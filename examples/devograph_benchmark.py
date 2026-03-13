#!/usr/bin/env python3
"""DevoGraph Benchmark: C. elegans embryogenesis with hgx.

End-to-end demonstration of hypergraph neural networks for
modeling developmental dynamics, targeting the OpenWorm DevoGraph project
(OpenWorm/DevoWorm GSoC).

Three benchmark tasks on synthetic C. elegans-like embryogenesis data:
  A. Cell type classification with HGNNStack
  B. Trajectory prediction with HypergraphNeuralODE (via LatentHypergraphODE)
  C. Topology-aware development with dynamic topology operations

Usage:
    uv run python examples/devograph_benchmark.py
    uv run python examples/devograph_benchmark.py --no-viz --epochs 200
    uv run python examples/devograph_benchmark.py --seed 123 --epochs 50

Requires: pip install hgx[dynamics] optax matplotlib networkx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import optax
from hgx._hypergraph import Hypergraph
from hgx._latent import LatentHypergraphODE


# ---------------------------------------------------------------------------
# Synthetic C. elegans-like data generation
# ---------------------------------------------------------------------------

# C. elegans founder cell names (AB lineage + others)
CELL_TYPE_NAMES = ["AB", "P1", "EMS", "P2", "ABa", "ABp", "MS", "E", "C", "P3"]


def build_knn_incidence(positions, k=4):
    """Build KNN hyperedges from spatial positions.

    Each node defines a hyperedge containing itself and its k-1 nearest
    neighbors, modeling local cell-cell communication neighborhoods.

    Args:
        positions: Array of shape (n, 3) with 3D cell coordinates.
        k: Neighborhood size (clamped to n).

    Returns:
        Incidence matrix of shape (n, n).
    """
    n = positions.shape[0]
    k = min(k, n)
    dists = jnp.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
    _, nn_idx = jax.lax.top_k(-dists, k)  # (n, k) nearest per node
    rows = nn_idx.flatten()
    cols = jnp.repeat(jnp.arange(n), k)
    return jnp.zeros((n, n)).at[rows, cols].set(1.0)


def generate_synthetic_embryo(
    num_cells=15,
    feature_dim=8,
    num_timesteps=10,
    num_types=4,
    k=4,
    *,
    key,
):
    """Generate synthetic embryo data mimicking C. elegans early development.

    Simulates an early embryo with:
    - 3D spatial positions undergoing radial expansion + noise
    - Cell-state feature vectors correlated with cell type
    - KNN-based hyperedges rebuilt at each timestep
    - Cell division events (new cells appear at later timesteps)

    Args:
        num_cells: Initial number of cells.
        feature_dim: Dimensionality of cell-state features.
        num_timesteps: Number of developmental time steps.
        num_types: Number of distinct cell types.
        k: KNN neighborhood size for hyperedge construction.
        key: PRNG key.

    Returns:
        Dictionary with keys:
            positions: (num_timesteps, num_cells, 3) spatial coordinates
            features: (num_timesteps, num_cells, feature_dim) cell states
            incidences: list of (num_cells, num_cells) incidence matrices
            labels: (num_cells,) integer cell type labels
            division_events: list of (timestep, parent_idx) tuples
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Assign cell types (roughly equal groups)
    labels = jnp.array([i % num_types for i in range(num_cells)], dtype=jnp.int32)

    # Initial 3D positions: clustered by type with some overlap
    pos0 = jax.random.normal(k1, (num_cells, 3)) * 0.5
    # Add type-dependent offset to create spatial structure
    type_centers = jax.random.normal(k4, (num_types, 3)) * 0.3
    pos0 = pos0 + type_centers[labels]

    # Initial features: type-correlated signal + noise
    type_signatures = jax.random.normal(k2, (num_types, feature_dim)) * 0.8
    noise = 0.2 * jax.random.normal(k3, (num_cells, feature_dim))
    feat0 = type_signatures[labels] + noise

    # Simulate developmental trajectory
    positions_list = [pos0]
    features_list = [feat0]

    for t in range(1, num_timesteps):
        kt = jax.random.fold_in(k3, t)
        kt1, kt2 = jax.random.split(kt)

        prev_pos = positions_list[-1]
        prev_feat = features_list[-1]

        # Radial drift (embryo expansion)
        r = jnp.linalg.norm(prev_pos, axis=1, keepdims=True).clip(1e-6)
        drift = 0.04 * prev_pos / r

        # Type-dependent migration bias
        type_bias = 0.02 * type_centers[labels]

        # Stochastic perturbation
        noise_pos = 0.03 * jax.random.normal(kt1, prev_pos.shape)
        new_pos = prev_pos + drift + type_bias + noise_pos

        # Feature evolution: slow drift toward type signature + diffusion
        feat_drift = 0.05 * (type_signatures[labels] - prev_feat)
        noise_feat = 0.02 * jax.random.normal(kt2, prev_feat.shape)
        new_feat = prev_feat + feat_drift + noise_feat

        positions_list.append(new_pos)
        features_list.append(new_feat)

    positions = jnp.stack(positions_list)  # (T, n, 3)
    features = jnp.stack(features_list)  # (T, n, d)
    incidences = [build_knn_incidence(positions[t], k=k) for t in range(num_timesteps)]

    # Record synthetic division events (for Task C)
    division_events = []
    kd = jax.random.fold_in(key, 999)
    for t in range(2, num_timesteps, 3):
        parent = int(jax.random.randint(jax.random.fold_in(kd, t), (), 0, num_cells))
        division_events.append((t, parent))

    return {
        "positions": positions,
        "features": features,
        "incidences": incidences,
        "labels": labels,
        "division_events": division_events,
        "num_types": num_types,
    }


# ---------------------------------------------------------------------------
# Task A: Cell type classification
# ---------------------------------------------------------------------------


def run_task_a(data, epochs, *, key):
    """Task A: Node classification (cell type prediction).

    Trains an HGNNStack with UniGCNConv layers to predict cell type
    from hypergraph-convolved features at the initial time step.
    """
    print("\n" + "=" * 60)
    print("Task A: Cell Type Classification")
    print("=" * 60)

    k1, k2 = jax.random.split(key)
    features = data["features"][0]  # initial features (n, d)
    labels = data["labels"]  # (n,)
    incidence = data["incidences"][0]
    num_types = data["num_types"]
    n, d = features.shape

    hg = hgx.from_incidence(incidence, node_features=features)

    # Model: 2-layer HGNNStack with readout
    model = hgx.HGNNStack(
        conv_dims=[(d, 32), (32, 16)],
        conv_cls=hgx.UniGCNConv,
        readout_dim=num_types,
        activation=jax.nn.relu,
        key=k1,
    )

    # Optimizer
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def cross_entropy(model, hg, labels):
        logits = model(hg, inference=True)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(labels, num_classes=num_types)
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    @eqx.filter_jit
    def train_step(model, opt_state, hg, labels):
        loss, grads = eqx.filter_value_and_grad(cross_entropy)(model, hg, labels)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    # Training loop
    losses_a = []
    t_start = time.time()

    for epoch in range(epochs):
        model, opt_state, loss = train_step(model, opt_state, hg, labels)
        losses_a.append(float(loss))
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logits = model(hg, inference=True)
            preds = jnp.argmax(logits, axis=-1)
            acc = float(jnp.mean(preds == labels))
            print(f"  Epoch {epoch + 1:4d}  loss={loss:.4f}  acc={acc:.1%}")

    elapsed = time.time() - t_start

    # Final evaluation
    logits = model(hg, inference=True)
    preds = jnp.argmax(logits, axis=-1)
    final_acc = float(jnp.mean(preds == labels))

    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Final accuracy: {final_acc:.1%}")

    return {"accuracy": final_acc, "losses": losses_a, "model": model}


# ---------------------------------------------------------------------------
# Task B: Trajectory prediction with Neural ODE
# ---------------------------------------------------------------------------


def run_task_b(data, epochs, *, key):
    """Task B: Trajectory prediction with LatentHypergraphODE.

    Trains a LatentHypergraphODE (encode -> Neural ODE -> decode) to
    predict cell feature evolution one step ahead.  Evaluates on the
    last 3 time steps via autoregressive rollout.
    """
    print("\n" + "=" * 60)
    print("Task B: Trajectory Prediction (Neural ODE)")
    print("=" * 60)

    k1, k2 = jax.random.split(key)

    positions = data["positions"]  # (T, n, 3)
    incidences = data["incidences"]
    num_steps = positions.shape[0]
    num_train = num_steps - 3  # hold out last 3 steps
    num_pairs = num_train - 1

    eval_steps = num_steps - num_train
    print(f"  Training pairs: {num_pairs}  |  Eval rollout: {eval_steps} steps")

    # Model: encode 3D positions -> 16D latent -> ODE -> decode
    model = LatentHypergraphODE(
        obs_dim=3,
        latent_dim=16,
        conv_cls=hgx.UniGCNConv,
        key=k1,
    )

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, hg, target):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            pred = m(hg, t0=0.0, t1=1.0)
            return jnp.mean((pred - target) ** 2)

        loss, grads = loss_fn(model)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    # Training loop: cycle through consecutive pairs
    losses_b = []
    t_start = time.time()

    for i in range(epochs):
        t = i % num_pairs
        hg_t = hgx.from_incidence(incidences[t], node_features=positions[t])
        target_t = positions[t + 1]

        model, opt_state, loss = train_step(model, opt_state, hg_t, target_t)
        losses_b.append(float(loss))

        if (i + 1) % max(1, epochs // 5) == 0:
            print(f"  Step {i + 1:4d}  loss={loss:.6f}")

    elapsed = time.time() - t_start
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluation: autoregressive rollout from the last training frame
    print("\n  Rollout evaluation (held-out frames):")
    pred_pos = positions[num_train - 1]
    pred_trajectory = [pred_pos]
    rollout_mses = []

    for t in range(num_train - 1, num_steps - 1):
        hg_t = hgx.from_incidence(incidences[t], node_features=pred_pos)
        pred_pos = model(hg_t, t0=0.0, t1=1.0)
        actual = positions[t + 1]
        mse = float(jnp.mean((pred_pos - actual) ** 2))
        rollout_mses.append(mse)
        pred_trajectory.append(pred_pos)
        print(f"    Frame {t + 1}: MSE = {mse:.6f}")

    pred_trajectory = jnp.stack(pred_trajectory)
    avg_mse = sum(rollout_mses) / len(rollout_mses)
    print(f"  Average rollout MSE: {avg_mse:.6f}")

    return {
        "mse": avg_mse,
        "losses": losses_b,
        "model": model,
        "pred_trajectory": pred_trajectory,
        "rollout_start": num_train - 1,
    }


# ---------------------------------------------------------------------------
# Task C: Topology-aware development (dynamic topology)
# ---------------------------------------------------------------------------


def run_task_c(data, *, key):
    """Task C: Topology-aware development with dynamic hypergraph operations.

    Demonstrates how hgx models cell division by dynamically adding
    nodes and hyperedges to a pre-allocated hypergraph. Shows that
    the topology grows over developmental steps while maintaining
    JIT compatibility.
    """
    print("\n" + "=" * 60)
    print("Task C: Topology-Aware Development (Dynamic Topology)")
    print("=" * 60)

    features = data["features"]
    positions = data["positions"]
    incidences = data["incidences"]
    division_events = data["division_events"]
    num_steps = features.shape[0]
    initial_cells = features.shape[1]

    # Estimate capacity: initial cells + one new cell per division event
    max_new_cells = len(division_events)
    max_nodes = initial_cells + max_new_cells + 5  # headroom
    max_edges = initial_cells + max_new_cells + 10

    print(f"  Initial cells: {initial_cells}")
    print(f"  Planned divisions: {len(division_events)}")
    print(f"  Pre-allocated capacity: {max_nodes} nodes, {max_edges} edges")

    # Build initial hypergraph and pre-allocate
    hg = hgx.from_incidence(
        incidences[0],
        node_features=features[0],
        positions=positions[0],
    )
    hg = hgx.preallocate(hg, max_nodes=max_nodes, max_edges=max_edges)

    k1, _ = jax.random.split(key)
    conv = hgx.UniGCNConv(in_dim=features.shape[2], out_dim=features.shape[2], key=k1)

    # Track growth
    cell_counts = [int(jnp.sum(hg.node_mask))]
    edge_counts = [int(jnp.sum(hg.edge_mask))]
    division_idx = 0

    print(f"\n  Step  0: {cell_counts[0]:3d} cells, {edge_counts[0]:3d} edges")

    for t in range(1, num_steps):
        # Run convolution to propagate signals (features evolve through the network)
        out = conv(hg)

        # Update features (use conv output masked by node_mask)
        new_features = jnp.where(
            hg.node_mask[:, None],
            out,
            hg.node_features,
        )
        hg = Hypergraph(
            node_features=new_features,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
        )

        # Check for cell division at this timestep
        if division_idx < len(division_events):
            div_time, parent_idx = division_events[division_idx]
            if t == div_time:
                # Daughter cell: perturbed copy of parent features
                parent_feat = hg.node_features[parent_idx]
                kt = jax.random.fold_in(key, t * 1000 + parent_idx)
                noise = jax.random.normal(kt, parent_feat.shape)
                daughter_feat = parent_feat + 0.1 * noise

                # Daughter joins same hyperedges as parent
                parent_edges = hg.incidence[parent_idx].astype(bool)

                hg = hgx.add_node(hg, features=daughter_feat, hyperedges=parent_edges)

                # Create a new hyperedge connecting parent and daughter
                # Find the daughter: highest-index active node after add_node
                active_indices = jnp.where(hg.node_mask, jnp.arange(max_nodes), -1)
                daughter_slot = jnp.max(active_indices)

                members = jnp.zeros(max_nodes, dtype=bool)
                members = members.at[parent_idx].set(True)
                members = members.at[daughter_slot].set(True)

                hg = hgx.add_hyperedge(hg, members=members)

                division_idx += 1

        n_cells = int(jnp.sum(hg.node_mask))
        n_edges = int(jnp.sum(hg.edge_mask))
        cell_counts.append(n_cells)
        edge_counts.append(n_edges)

        if t % max(1, (num_steps - 1) // 4) == 0 or t == num_steps - 1:
            print(f"  Step {t:2d}: {n_cells:3d} cells, {n_edges:3d} edges")

    final_cells = cell_counts[-1]
    final_edges = edge_counts[-1]
    print(f"\n  Growth: {initial_cells} -> {final_cells} cells over {num_steps} steps")
    print(f"  Edges:  {edge_counts[0]} -> {final_edges}")

    return {
        "initial_cells": initial_cells,
        "final_cells": final_cells,
        "cell_counts": cell_counts,
        "edge_counts": edge_counts,
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(results_a["losses"], linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Task A: Cell Type Classification Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results_b["losses"], linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Task B: Trajectory Prediction Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = fig_dir / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Figure 2: Spatial trajectory comparison ---
    positions = data["positions"]
    pred_traj = results_b["pred_trajectory"]
    num_cells = positions.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.tab10.colors

    # XY projection: actual vs predicted
    ax = axes[0]
    for c in range(min(5, num_cells)):
        ax.plot(
            positions[:, c, 0],
            positions[:, c, 1],
            "o-",
            color=colors[c],
            markersize=3,
            linewidth=1,
            label=f"Cell {c} actual",
        )
        ax.plot(
            pred_traj[:, c, 0],
            pred_traj[:, c, 1],
            "s--",
            color=colors[c],
            markersize=2,
            alpha=0.6,
            linewidth=1,
            label=f"Cell {c} pred",
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("XY Trajectory: Actual vs Predicted")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # XZ projection
    ax = axes[1]
    for c in range(min(5, num_cells)):
        ax.plot(
            positions[:, c, 0],
            positions[:, c, 2],
            "o-",
            color=colors[c],
            markersize=3,
            linewidth=1,
        )
        ax.plot(
            pred_traj[:, c, 0],
            pred_traj[:, c, 2],
            "s--",
            color=colors[c],
            markersize=2,
            alpha=0.6,
            linewidth=1,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("XZ Trajectory: Actual vs Predicted")
    ax.grid(True, alpha=0.3)

    # Topology growth
    ax = axes[2]
    steps = list(range(len(results_c["cell_counts"])))
    ax.plot(steps, results_c["cell_counts"], "o-", label="Cells", markersize=4)
    ax.plot(steps, results_c["edge_counts"], "s-", label="Hyperedges", markersize=4)
    ax.set_xlabel("Developmental Step")
    ax.set_ylabel("Count")
    ax.set_title("Task C: Topology Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = fig_dir / "trajectory_and_growth.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Figure 3: Hypergraph snapshots at different time steps ---
    try:
        import networkx  # noqa: F401 -- needed by draw_hypergraph

        snapshot_times = [0, len(data["incidences"]) // 2, len(data["incidences"]) - 1]
        ncols = len(snapshot_times)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

        for idx, t in enumerate(snapshot_times):
            # Build a 2D projection for visualization
            pos_2d = data["positions"][t][:, :2]
            feat = data["features"][t]
            inc = data["incidences"][t]
            hg_snap = hgx.from_incidence(
                inc,
                node_features=feat,
                positions=pos_2d,
            )

            label_arr = data["labels"]
            type_names = [
                CELL_TYPE_NAMES[int(l) % len(CELL_TYPE_NAMES)] for l in label_arr
            ]
            hgx.draw_hypergraph(
                hg_snap,
                ax=axes[idx],
                node_labels=type_names,
                title=f"Embryo at t={t}",
                node_size=300,
            )

        fig.tight_layout()
        path = fig_dir / "hypergraph_snapshots.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")
    except Exception as exc:
        print(f"  Skipping hypergraph snapshots: {exc}")

    # --- Figure 4: Incidence matrix evolution ---
    try:
        snapshot_times = [0, len(data["incidences"]) // 2, len(data["incidences"]) - 1]
        ncols = len(snapshot_times)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

        for idx, t in enumerate(snapshot_times):
            feat = data["features"][t]
            inc = data["incidences"][t]
            hg_snap = hgx.from_incidence(inc, node_features=feat)
            hgx.draw_incidence(hg_snap, ax=axes[idx], title=f"Incidence at t={t}")

        fig.tight_layout()
        path = fig_dir / "incidence_evolution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")
    except Exception as exc:
        print(f"  Skipping incidence plots: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="DevoGraph Benchmark: C. elegans embryogenesis with hgx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/devograph_benchmark.py\n"
            "  python examples/devograph_benchmark.py --no-viz --epochs 200\n"
            "  python examples/devograph_benchmark.py --seed 123\n"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for Tasks A and B (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (useful for headless / CI runs)",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=15,
        help="Initial number of cells (default: 15)",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=10,
        help="Number of developmental time steps (default: 10)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DevoGraph Benchmark")
    print("  C. elegans embryogenesis with hgx hypergraph networks")
    print("  Target: OpenWorm/DevoWorm GSoC DevoGraph project")
    print("=" * 60)

    key = jax.random.PRNGKey(args.seed)
    k_data, k_a, k_b, k_c = jax.random.split(key, 4)

    # --- Generate synthetic embryo data ---
    print("\nGenerating synthetic C. elegans embryo data...")
    t0 = time.time()
    data = generate_synthetic_embryo(
        num_cells=args.num_cells,
        feature_dim=8,
        num_timesteps=args.num_timesteps,
        num_types=4,
        k=4,
        key=k_data,
    )
    print(
        f"  Cells: {args.num_cells}  |  Timesteps: {args.num_timesteps}  "
        f"|  Feature dim: 8  |  Types: 4"
    )
    print(f"  Positions shape: {data['positions'].shape}")
    print(f"  Features shape:  {data['features'].shape}")
    print(f"  Division events: {data['division_events']}")
    print(f"  Data generation: {time.time() - t0:.2f}s")

    # --- Run benchmark tasks ---
    total_start = time.time()

    results_a = run_task_a(data, epochs=args.epochs, key=k_a)
    results_b = run_task_b(data, epochs=args.epochs, key=k_b)
    results_c = run_task_c(data, key=k_c)

    total_elapsed = time.time() - total_start

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  DevoGraph Benchmark Results")
    print("=" * 60)
    acc = results_a["accuracy"]
    print(f"  Task A (Cell Type Classification): accuracy = {acc:.1%}")
    print(f"  Task B (Trajectory Prediction):    MSE = {results_b['mse']:.6f}")
    print(
        f"  Task C (Topology Growth):          cells: "
        f"{results_c['initial_cells']} -> {results_c['final_cells']} "
        f"over {args.num_timesteps} steps"
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
