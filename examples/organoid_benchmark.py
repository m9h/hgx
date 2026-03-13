#!/usr/bin/env python3
"""Organoid Regulome Benchmark: cerebral organoid gene regulation with hgx.

End-to-end demonstration of hypergraph neural networks for modeling
gene regulatory networks, inspired by the Flauger et al. 2022 cerebral
organoid regulome study.

Three benchmark tasks on synthetic regulome data:
  A. Module detection with HGNNStack + UniGATConv
  B. Fate trajectory with PoincareHypergraphConv + HypergraphNeuralODE
  C. Perturbation prediction (held-out TF knockouts)

Usage:
    uv run python examples/organoid_benchmark.py
    uv run python examples/organoid_benchmark.py --no-viz --epochs 40
    uv run python examples/organoid_benchmark.py --seed 123

Requires: pip install hgx[dynamics] optax matplotlib
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hgx._latent import LatentHypergraphODE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FATE_NAMES = ["Cortical", "MGE", "LGE", "CGE"]
NUM_GENES = 200
NUM_TFS = 20
NUM_MASTER = 3
NUM_INTERMEDIATE = 7
NUM_MODULES = 15
NUM_FATES = 4
NUM_PSEUDOTIME = 10
FEATURE_DIM = 8


# ---------------------------------------------------------------------------
# Synthetic regulome data generation
# ---------------------------------------------------------------------------


def generate_synthetic_regulome(*, key):
    """Generate synthetic cerebral organoid regulome data.

    Creates a hierarchical gene regulatory network with 200 genes
    (20 TFs, 180 targets), 15 regulatory modules, 4 cell fates,
    10 pseudotime steps, and 5 TF knockout perturbations.

    Returns:
        Dictionary with incidence matrix, expression trajectories,
        module labels, fate assignments, and perturbation data.
    """
    seed_val = int(jax.random.randint(key, (), 0, 2**30))
    rng = np.random.RandomState(seed_val)

    # --- Build regulatory modules (hyperedges) ---
    # Master TFs (0-2): large modules (~25 targets each)
    # Intermediate TFs (3-9): medium modules (~15 targets each)
    # Downstream TFs (10-14): small modules (~10 targets each)
    incidence = np.zeros((NUM_GENES, NUM_MODULES), dtype=np.float32)
    module_labels = np.full(NUM_GENES, -1, dtype=np.int32)

    target_pool = list(range(NUM_TFS, NUM_GENES))
    rng.shuffle(target_pool)

    idx = 0
    for m in range(NUM_MODULES):
        if m < NUM_MASTER:
            tf, n_targets = m, 25
        elif m < NUM_MASTER + NUM_INTERMEDIATE:
            tf, n_targets = m, 15
        else:
            tf = (m - NUM_MASTER - NUM_INTERMEDIATE) + 10
            n_targets = 10

        incidence[tf, m] = 1.0
        module_labels[tf] = m

        end = min(idx + n_targets, len(target_pool))
        for t_idx in range(idx, end):
            g = target_pool[t_idx]
            incidence[g, m] = 1.0
            if module_labels[g] == -1:
                module_labels[g] = m
        idx = end

    # Assign remaining unassigned genes to random modules
    for g in range(NUM_GENES):
        if module_labels[g] == -1:
            m = rng.randint(NUM_MODULES)
            incidence[g, m] = 1.0
            module_labels[g] = m

    # Hierarchical cross-module connections:
    # Master TFs participate in intermediate modules
    for m_tf in range(NUM_MASTER):
        for int_mod in range(NUM_MASTER, NUM_MASTER + 3):
            incidence[m_tf, int_mod] = 1.0
    # Intermediate TFs participate in some downstream modules
    for i_tf in range(NUM_MASTER, NUM_MASTER + NUM_INTERMEDIATE):
        for d_mod in range(NUM_MASTER + NUM_INTERMEDIATE, NUM_MODULES):
            if rng.random() < 0.3:
                incidence[i_tf, d_mod] = 1.0

    # Gene-to-fate mapping based on module
    module_to_fate = np.array(
        [min(m * NUM_FATES // NUM_MODULES, NUM_FATES - 1) for m in range(NUM_MODULES)]
    )
    gene_fates = module_to_fate[module_labels]

    # --- Expression trajectories over pseudotime ---
    k1, k2, k3 = jax.random.split(key, 3)
    base_expr = jax.random.normal(k1, (NUM_GENES, FEATURE_DIM)) * 0.3

    fate_sigs = jnp.stack(
        [jax.random.normal(jax.random.fold_in(k2, f), (FEATURE_DIM,))
         for f in range(NUM_FATES)]
    )

    trajectories = []
    for t in range(NUM_PSEUDOTIME):
        frac = t / max(NUM_PSEUDOTIME - 1, 1)
        fate_bias = fate_sigs[jnp.array(gene_fates)] * frac
        noise = 0.1 * jax.random.normal(
            jax.random.fold_in(k3, t), (NUM_GENES, FEATURE_DIM)
        )
        trajectories.append(base_expr + fate_bias + noise)
    trajectories = jnp.stack(trajectories)  # (T, N, D)

    # --- TF knockout perturbations (3 train, 2 test) ---
    perturb_tfs = [0, 3, 7, 12, 15]
    perturbation_data = {}
    for i, tf_idx in enumerate(perturb_tfs):
        tf_modules = incidence[tf_idx]
        affected = (incidence @ tf_modules) > 0

        effect = np.zeros(NUM_GENES, dtype=np.float32)
        n_affected = int(np.sum(affected))
        effect[affected] = -0.5 + rng.randn(n_affected) * 0.2
        effect[tf_idx] = -1.0

        perturbation_data[tf_idx] = {
            "effect": jnp.array(effect),
            "affected_mask": jnp.array(affected),
            "original_fate": int(gene_fates[tf_idx]),
            "shifted_fate": (int(gene_fates[tf_idx]) + 1) % NUM_FATES,
        }

    return {
        "incidence": jnp.array(incidence),
        "trajectories": trajectories,
        "module_labels": jnp.array(module_labels),
        "gene_fates": jnp.array(gene_fates),
        "perturbation_data": perturbation_data,
        "train_perturb_tfs": perturb_tfs[:3],
        "test_perturb_tfs": perturb_tfs[3:],
    }


# ---------------------------------------------------------------------------
# Task A: Module detection
# ---------------------------------------------------------------------------


def run_task_a(data, epochs, *, key):
    """Task A: Regulatory module detection with HGNNStack + UniGATConv.

    Classifies genes into 15 regulatory modules. Reports macro F1 and
    correlation between learned attention weights and ground-truth incidence.
    """
    print("\n" + "=" * 60)
    print("Task A: Regulatory Module Detection")
    print("=" * 60)

    k1, _ = jax.random.split(key)
    features = data["trajectories"][-1]
    labels = data["module_labels"]
    incidence = data["incidence"]
    hg = hgx.from_incidence(incidence, node_features=features)

    model = hgx.HGNNStack(
        conv_dims=[(FEATURE_DIM, 32), (32, 16)],
        conv_cls=hgx.UniGATConv,
        readout_dim=NUM_MODULES,
        activation=jax.nn.relu,
        key=k1,
    )

    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, hg, labels):
        logits = model(hg, inference=True)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, num_classes=NUM_MODULES)
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    @eqx.filter_jit
    def step(model, opt_state, hg, labels):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, hg, labels)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt_state, loss

    losses = []
    t_start = time.time()
    for epoch in range(epochs):
        model, opt_state, loss = step(model, opt_state, hg, labels)
        losses.append(float(loss))
        if (epoch + 1) % max(1, epochs // 5) == 0:
            preds = jnp.argmax(model(hg, inference=True), axis=-1)
            acc = float(jnp.mean(preds == labels))
            print(f"  Epoch {epoch+1:4d}  loss={loss:.4f}  acc={acc:.1%}")

    elapsed = time.time() - t_start
    preds = jnp.argmax(model(hg, inference=True), axis=-1)

    # Macro F1
    f1s = []
    for c in range(NUM_MODULES):
        tp = float(jnp.sum((preds == c) & (labels == c)))
        fp = float(jnp.sum((preds == c) & (labels != c)))
        fn = float(jnp.sum((preds != c) & (labels == c)))
        prec = tp / max(tp + fp, 1e-8)
        rec = tp / max(tp + fn, 1e-8)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    macro_f1 = float(np.mean(f1s))

    # Attention-incidence correlation from first UniGATConv layer
    conv0 = model.convs[0]
    H = hg._masked_incidence()
    x_proj = jax.vmap(conv0.linear)(features)
    out_dim = x_proj.shape[-1]
    if conv0.normalize:
        d_e = jnp.sum(H, axis=0, keepdims=True)
        e_repr = (H * jnp.where(d_e > 0, 1.0 / d_e, 0.0)).T @ x_proj
    else:
        e_repr = H.T @ x_proj
    v_sc = x_proj @ conv0.attn[:out_dim]
    e_sc = e_repr @ conv0.attn[out_dim:]
    raw = v_sc[:, None] + e_sc[None, :]
    raw = jnp.where(raw >= 0, raw, conv0.negative_slope * raw)
    raw = jnp.where(H > 0, raw, -1e9)
    attn = jax.nn.softmax(raw, axis=1) * H
    attn_corr = float(
        np.corrcoef(np.array(attn).ravel(), np.array(incidence).ravel())[0, 1]
    )

    print(
        f"\n  Time: {elapsed:.1f}s  |  Macro F1: {macro_f1:.3f}"
        f"  |  Attn corr: {attn_corr:.3f}"
    )
    return {
        "macro_f1": macro_f1,
        "attn_corr": attn_corr,
        "losses": losses,
        "preds": preds,
    }


# ---------------------------------------------------------------------------
# Task B: Fate trajectory
# ---------------------------------------------------------------------------


def _poincare_dist(x, y, c=1.0):
    """Poincare ball distance between two points."""
    diff_sq = jnp.sum((x - y) ** 2)
    x_sq, y_sq = jnp.sum(x ** 2), jnp.sum(y ** 2)
    denom = jnp.maximum((1 - c * x_sq) * (1 - c * y_sq), 1e-8)
    return jnp.arccosh(jnp.maximum(1 + 2 * c * diff_sq / denom, 1.0 + 1e-7))


def compute_gromov_delta(embeddings, c=1.0, n_samples=25):
    """Estimate Gromov delta-hyperbolicity from Poincare embeddings.

    Uses the four-point condition on a random subsample.
    Lower delta indicates more hyperbolic (tree-like) structure.
    """
    n = embeddings.shape[0]
    idx = np.random.choice(n, min(n_samples, n), replace=False)
    pts = embeddings[idx]
    k = len(idx)

    # Pairwise distance matrix
    dists = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = float(_poincare_dist(pts[i], pts[j], c))
            dists[i, j] = dists[j, i] = d

    # Four-point condition: delta = max over quadruples of (S_max - S_mid) / 2
    delta = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            for ci in range(b + 1, k):
                for di in range(ci + 1, k):
                    s1 = dists[a, b] + dists[ci, di]
                    s2 = dists[a, ci] + dists[b, di]
                    s3 = dists[a, di] + dists[b, ci]
                    sums = sorted([s1, s2, s3], reverse=True)
                    delta = max(delta, (sums[0] - sums[1]) / 2)

    return delta


def run_task_b(data, epochs, *, key):
    """Task B: Fate trajectory with PoincareHypergraphConv + HypergraphNeuralODE.

    Trains a LatentHypergraphODE with Poincare-ball convolution to predict
    gene expression evolution one step ahead. Reports trajectory MSE and
    Gromov delta-hyperbolicity of the learned Poincare embeddings.
    """
    print("\n" + "=" * 60)
    print("Task B: Fate Trajectory (Poincare ODE)")
    print("=" * 60)

    k1, _ = jax.random.split(key)
    trajs = data["trajectories"]  # (T, N, D)
    incidence = data["incidence"]
    num_train = NUM_PSEUDOTIME - 3
    num_pairs = num_train - 1
    eval_steps = NUM_PSEUDOTIME - num_train
    print(f"  Training pairs: {num_pairs}  |  Eval rollout: {eval_steps} steps")

    model = LatentHypergraphODE(
        obs_dim=FEATURE_DIM,
        latent_dim=8,
        conv_cls=hgx.PoincareHypergraphConv,
        key=k1,
    )

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, hg, target):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            pred = m(hg, t0=0.0, t1=1.0)
            return jnp.mean((pred - target) ** 2)

        loss, grads = loss_fn(model)
        updates, new_opt = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), new_opt, loss

    losses = []
    t_start = time.time()
    for i in range(epochs):
        t = i % num_pairs
        hg_t = hgx.from_incidence(incidence, node_features=trajs[t])
        model, opt_state, loss = step(model, opt_state, hg_t, trajs[t + 1])
        losses.append(float(loss))
        if (i + 1) % max(1, epochs // 5) == 0:
            print(f"  Step {i+1:4d}  loss={loss:.6f}")

    elapsed = time.time() - t_start
    print(f"  Training time: {elapsed:.1f}s")

    # Rollout evaluation on held-out frames
    print("\n  Rollout evaluation (held-out frames):")
    pred_feat = trajs[num_train - 1]
    rollout_mses = []
    for t in range(num_train - 1, NUM_PSEUDOTIME - 1):
        hg_t = hgx.from_incidence(incidence, node_features=pred_feat)
        pred_feat = model(hg_t, t0=0.0, t1=1.0)
        mse = float(jnp.mean((pred_feat - trajs[t + 1]) ** 2))
        rollout_mses.append(mse)
        print(f"    Frame {t+1}: MSE = {mse:.6f}")

    avg_mse = float(np.mean(rollout_mses))

    # Gromov delta-hyperbolicity on Poincare embeddings
    print("\n  Computing Gromov delta-hyperbolicity...")
    final_features = trajs[-1]
    z = jax.vmap(model.encoder)(final_features)
    c_val = float(jnp.abs(model.dynamics.drift.conv.c) + 1e-6)
    # Project into Poincare ball
    z_np = np.array(z)
    max_norm = 1.0 / np.sqrt(c_val) - 1e-5
    norms = np.linalg.norm(z_np, axis=-1, keepdims=True)
    z_ball = z_np * np.minimum(max_norm / np.maximum(norms, 1e-6), 1.0)

    gromov_delta = compute_gromov_delta(jnp.array(z_ball), c=c_val)

    print(f"\n  Avg rollout MSE: {avg_mse:.6f}")
    print(f"  Gromov delta: {gromov_delta:.4f}")

    return {
        "mse": avg_mse,
        "gromov_delta": gromov_delta,
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# Task C: Perturbation prediction
# ---------------------------------------------------------------------------


class _PerturbPredictor(eqx.Module):
    """Predicts gene expression changes from a TF knockout."""

    conv: hgx.UniGCNConv
    readout: eqx.nn.Linear

    def __init__(self, in_dim, hidden_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.conv = hgx.UniGCNConv(in_dim, hidden_dim, key=k1)
        self.readout = eqx.nn.Linear(hidden_dim, 1, key=k2)

    def __call__(self, hg):
        x = jax.nn.relu(self.conv(hg))
        return jax.vmap(self.readout)(x).squeeze(-1)


def run_task_c(data, epochs, *, key):
    """Task C: Perturbation prediction (TF knockout effects).

    Trains on 3 TF knockouts to predict gene expression changes, then
    evaluates on 2 held-out knockouts. Reports Pearson correlation and
    fate shift accuracy.
    """
    print("\n" + "=" * 60)
    print("Task C: Perturbation Prediction")
    print("=" * 60)

    k1, _ = jax.random.split(key)
    base_features = data["trajectories"][-1]  # (N, D)
    incidence = data["incidence"]
    perturb_data = data["perturbation_data"]
    train_tfs = data["train_perturb_tfs"]
    test_tfs = data["test_perturb_tfs"]
    print(f"  Train KOs: TFs {train_tfs}  |  Test KOs: TFs {test_tfs}")

    model = _PerturbPredictor(FEATURE_DIM, 32, key=k1)
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, hg, target_effect):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            pred = m(hg)
            return jnp.mean((pred - target_effect) ** 2)

        loss, grads = loss_fn(model)
        updates, new_opt = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), new_opt, loss

    # Pre-build training data: perturbed hypergraphs and effect vectors
    train_hgs = []
    train_effects = []
    for tf_idx in train_tfs:
        perturbed = base_features.at[tf_idx].set(jnp.zeros(FEATURE_DIM))
        train_hgs.append(hgx.from_incidence(incidence, node_features=perturbed))
        train_effects.append(perturb_data[tf_idx]["effect"])

    losses = []
    t_start = time.time()
    for i in range(epochs):
        t = i % len(train_tfs)
        model, opt_state, loss = step(model, opt_state, train_hgs[t], train_effects[t])
        losses.append(float(loss))
        if (i + 1) % max(1, epochs // 5) == 0:
            print(f"  Step {i+1:4d}  loss={loss:.6f}")

    elapsed = time.time() - t_start
    print(f"  Training time: {elapsed:.1f}s")

    # Test evaluation on held-out knockouts
    print("\n  Test KO evaluation:")
    correlations = []
    fate_accs = []
    for tf_idx in test_tfs:
        perturbed = base_features.at[tf_idx].set(jnp.zeros(FEATURE_DIM))
        hg_test = hgx.from_incidence(incidence, node_features=perturbed)
        pred_effect = model(hg_test)
        actual_effect = perturb_data[tf_idx]["effect"]
        affected = perturb_data[tf_idx]["affected_mask"]

        # Pearson correlation
        corr = float(
            np.corrcoef(np.array(pred_effect), np.array(actual_effect))[0, 1]
        )
        correlations.append(corr)

        # Fate shift accuracy: correctly predict which genes are affected
        pred_affected = jnp.abs(pred_effect) > 0.2
        acc = float(jnp.mean(pred_affected == affected))
        fate_accs.append(acc)
        print(f"    TF {tf_idx}: corr={corr:.3f}  fate_shift_acc={acc:.1%}")

    avg_corr = float(np.mean(correlations))
    avg_fate_acc = float(np.mean(fate_accs))

    print(f"\n  Avg correlation: {avg_corr:.3f}")
    print(f"  Avg fate shift accuracy: {avg_fate_acc:.1%}")

    return {
        "correlation": avg_corr,
        "fate_accuracy": avg_fate_acc,
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_plots(data, results_a, results_b, results_c, fig_dir):
    """Generate and save benchmark visualization figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available -- skipping visualization.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Training loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, losses, title in zip(
        axes,
        [results_a["losses"], results_b["losses"], results_c["losses"]],
        ["A: Module Detection", "B: Fate Trajectory", "C: Perturbation"],
    ):
        ax.plot(losses, linewidth=1.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = fig_dir / "organoid_losses.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Figure 2: Module detection results ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-module accuracy
    ax = axes[0]
    labels_np = np.array(data["module_labels"])
    preds_np = np.array(results_a["preds"])
    per_mod_acc = []
    for m in range(NUM_MODULES):
        mask = labels_np == m
        if np.sum(mask) > 0:
            per_mod_acc.append(float(np.mean(preds_np[mask] == m)))
        else:
            per_mod_acc.append(0.0)
    ax.bar(range(NUM_MODULES), per_mod_acc, color="steelblue")
    ax.set_xlabel("Module")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Module Classification Accuracy")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Incidence structure
    ax = axes[1]
    ax.imshow(np.array(data["incidence"]).T, aspect="auto", cmap="Blues")
    ax.set_xlabel("Gene Index")
    ax.set_ylabel("Module")
    ax.set_title("Regulatory Module Incidence")

    fig.tight_layout()
    path = fig_dir / "organoid_modules.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Figure 3: Expression landscape + Gromov delta ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Expression PCA at different pseudotime points
    ax = axes[0]
    trajs = np.array(data["trajectories"])
    fates = np.array(data["gene_fates"])
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for t_idx in [0, NUM_PSEUDOTIME // 2, NUM_PSEUDOTIME - 1]:
        expr = trajs[t_idx]
        centered = expr - expr.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        pc = centered @ Vt[:2].T
        alpha = 0.3 + 0.7 * (t_idx / (NUM_PSEUDOTIME - 1))
        for f in range(NUM_FATES):
            mask = fates == f
            label = FATE_NAMES[f] if t_idx == NUM_PSEUDOTIME - 1 else None
            ax.scatter(
                pc[mask, 0], pc[mask, 1],
                c=colors[f], alpha=alpha, s=10, label=label,
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Gene Expression PCA (light=early, dark=late)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Gromov delta summary
    ax = axes[1]
    ax.text(
        0.5, 0.6, f"Gromov delta = {results_b['gromov_delta']:.4f}",
        ha="center", va="center", fontsize=20, transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.4, f"Trajectory MSE = {results_b['mse']:.6f}",
        ha="center", va="center", fontsize=16, transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.2, "(lower delta = more hyperbolic)",
        ha="center", va="center", fontsize=11, color="gray",
        transform=ax.transAxes,
    )
    ax.set_title("Poincare Embedding Quality")
    ax.axis("off")

    fig.tight_layout()
    path = fig_dir / "organoid_trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Organoid Regulome Benchmark: cerebral organoid gene regulation with hgx"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/organoid_benchmark.py\n"
            "  python examples/organoid_benchmark.py --no-viz --epochs 40\n"
            "  python examples/organoid_benchmark.py --seed 123\n"
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Training epochs per task (default: 40)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization (useful for headless / CI runs)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Organoid Regulome Benchmark")
    print("  Cerebral organoid gene regulation with hgx")
    print("  Inspired by Flauger et al. 2022")
    print("=" * 60)

    key = jax.random.PRNGKey(args.seed)
    k_data, k_a, k_b, k_c = jax.random.split(key, 4)

    # --- Generate synthetic data ---
    print("\nGenerating synthetic regulome data...")
    t0 = time.time()
    data = generate_synthetic_regulome(key=k_data)
    print(
        f"  {NUM_GENES} genes ({NUM_TFS} TFs, {NUM_GENES - NUM_TFS} targets)"
    )
    print(
        f"  {NUM_MODULES} modules  |  {NUM_FATES} fates  |  "
        f"{NUM_PSEUDOTIME} timepoints"
    )
    print(f"  Incidence shape: {data['incidence'].shape}")
    print(f"  Trajectories shape: {data['trajectories'].shape}")
    print(f"  Data generation: {time.time() - t0:.2f}s")

    # --- Run benchmark tasks ---
    total_start = time.time()
    results_a = run_task_a(data, epochs=args.epochs, key=k_a)
    results_b = run_task_b(data, epochs=args.epochs, key=k_b)
    results_c = run_task_c(data, epochs=args.epochs, key=k_c)
    total_elapsed = time.time() - total_start

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Organoid Regulome Benchmark Results")
    print("=" * 60)
    print(
        f"  Task A: Macro F1 = {results_a['macro_f1']:.3f}"
        f"  |  Attn corr = {results_a['attn_corr']:.3f}"
    )
    print(
        f"  Task B: Traj MSE = {results_b['mse']:.6f}"
        f"  |  Gromov delta = {results_b['gromov_delta']:.4f}"
    )
    print(
        f"  Task C: Corr = {results_c['correlation']:.3f}"
        f"  |  Fate acc = {results_c['fate_accuracy']:.1%}"
    )
    print(f"\n  Total benchmark time: {total_elapsed:.1f}s")
    print("=" * 60)

    # --- Visualization ---
    if not args.no_viz:
        print("\nGenerating visualizations...")
        fig_dir = Path(__file__).parent / "figures"
        generate_plots(data, results_a, results_b, results_c, fig_dir)
        print("Done.")
    else:
        print("\nVisualization skipped (--no-viz).")


if __name__ == "__main__":
    main()
