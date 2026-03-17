#!/usr/bin/env python3
"""Benchmark on standard citation datasets (Cora, Citeseer, Pubmed).

Downloads Planetoid citation datasets, constructs 1-hop neighborhood
hypergraphs, and evaluates HGNNConv / UniGCNConv with configurable
architecture and training options.

Usage:
    python scripts/benchmark_standard.py --dataset cora
    python scripts/benchmark_standard.py --dataset all --num-seeds 5
    python scripts/benchmark_standard.py --dataset citeseer --conv HGNNConv --residual --layer-norm
    python scripts/benchmark_standard.py --dataset pubmed --sparse
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import hgx

# ---------------------------------------------------------------------------
# Dataset loading (Planetoid via torch_geometric or manual download)
# ---------------------------------------------------------------------------

def load_planetoid(name: str, data_dir: str = "~/.hgx_data"):
    """Load a Planetoid citation dataset.

    Returns:
        node_features: (n, d) float32
        labels: (n,) int32
        train_mask, val_mask, test_mask: (n,) bool
        edge_list: list of sets (1-hop neighborhoods as hyperedges)
    """
    data_dir = Path(data_dir).expanduser() / name
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=str(data_dir.parent), name=name)
        data = dataset[0]

        features = np.array(data.x, dtype=np.float32)
        labels = np.array(data.y, dtype=np.int32)
        train_mask = np.array(data.train_mask, dtype=bool)
        val_mask = np.array(data.val_mask, dtype=bool)
        test_mask = np.array(data.test_mask, dtype=bool)
        edge_index = np.array(data.edge_index, dtype=np.int64)
    except ImportError:
        raise ImportError(
            "torch_geometric is required for dataset loading. "
            "Install with: uv pip install torch-geometric torch"
        )

    # Build 1-hop neighborhood hyperedges: for each node, its neighbors + self
    n = features.shape[0]
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        neighbors[src].add(dst)

    edge_list = []
    for node in range(n):
        he = neighbors[node] | {node}  # include self
        if len(he) >= 2:  # skip trivial singleton hyperedges
            edge_list.append(he)

    return (
        jnp.array(features),
        jnp.array(labels),
        jnp.array(train_mask),
        jnp.array(val_mask),
        jnp.array(test_mask),
        edge_list,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

CONV_MAP = {
    "UniGCNConv": hgx.UniGCNConv,
    "HGNNConv": hgx.HGNNConv,
}

SPARSE_CONV_MAP = {
    "UniGCNConv": hgx.UniGCNSparseConv,
    "HGNNConv": hgx.HGNNSparseConv,
}


def build_model(
    conv_name: str,
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    residual: bool = False,
    layer_norm: bool = False,
    initial_alpha: float | None = None,
    sparse: bool = False,
    *,
    key,
):
    """Build an HGNNStack model."""
    conv_cls = (SPARSE_CONV_MAP if sparse else CONV_MAP)[conv_name]
    dims = [(in_dim, hidden_dim)] + [(hidden_dim, hidden_dim)] * (num_layers - 1)
    return hgx.HGNNStack(
        conv_dims=dims,
        conv_cls=conv_cls,
        readout_dim=num_classes,
        dropout_rate=dropout,
        residual=residual,
        layer_norm=layer_norm,
        initial_alpha=initial_alpha,
        key=key,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cross_entropy_loss(model, hg, labels, train_mask, key):
    logits = model(hg, key=key, inference=False)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, log_probs.shape[-1])
    per_node = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.sum(per_node * train_mask) / jnp.sum(train_mask)


@eqx.filter_jit
def train_step(model, hg, labels, train_mask, opt_state, optimizer, key):
    loss, grads = eqx.filter_value_and_grad(cross_entropy_loss)(
        model, hg, labels, train_mask, key
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def evaluate(model, hg, labels, mask):
    logits = model(hg, inference=True)
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum((preds == labels) * mask)
    return correct / jnp.sum(mask)


def train_and_eval(
    model,
    hg,
    labels,
    train_mask,
    val_mask,
    test_mask,
    *,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    num_epochs: int = 200,
    patience: int = 30,
    key,
    verbose: bool = True,
):
    """Train with early stopping on validation accuracy."""
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_val_acc = 0.0
    best_model = model
    patience_counter = 0

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        model, opt_state, loss = train_step(
            model, hg, labels, train_mask, opt_state, optimizer, subkey
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_acc = float(evaluate(model, hg, labels, val_mask))
            train_acc = float(evaluate(model, hg, labels, train_mask))
            if verbose:
                print(
                    f"  Epoch {epoch+1:3d}  loss={float(loss):.4f}  "
                    f"train={train_acc:.2%}  val={val_acc:.2%}"
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 10
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

    test_acc = float(evaluate(best_model, hg, labels, test_mask))
    return test_acc, best_val_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = ["Cora", "CiteSeer", "PubMed"]

PUBLISHED_BASELINES = {
    "Cora": 79.39,
    "CiteSeer": 72.01,
    "PubMed": 86.44,
}


def run_benchmark(args):
    datasets = DATASETS if args.dataset.lower() == "all" else [args.dataset]

    results = {}
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        features, labels, train_mask, val_mask, test_mask, edge_list = load_planetoid(ds_name)
        n, d = features.shape
        num_classes = int(labels.max()) + 1
        print(f"  Nodes: {n}  Features: {d}  Classes: {num_classes}")
        print(f"  Train: {int(train_mask.sum())}  Val: {int(val_mask.sum())}  Test: {int(test_mask.sum())}")
        print(f"  Hyperedges: {len(edge_list)}")

        # Build hypergraph
        if args.sparse:
            hg = hgx.from_edge_list_sparse(edge_list, node_features=features, num_nodes=n)
        else:
            hg = hgx.from_edge_list(edge_list, num_nodes=n, node_features=features)

        test_accs = []
        for seed in range(args.num_seeds):
            key = jax.random.PRNGKey(seed)
            model = build_model(
                conv_name=args.conv,
                in_dim=d,
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                residual=args.residual,
                layer_norm=args.layer_norm,
                initial_alpha=args.initial_alpha,
                sparse=args.sparse,
                key=key,
            )

            verbose = args.num_seeds == 1
            if not verbose:
                print(f"  Seed {seed}...", end=" ", flush=True)

            t0 = time.time()
            test_acc, val_acc = train_and_eval(
                model, hg, labels, train_mask, val_mask, test_mask,
                lr=args.lr, weight_decay=args.weight_decay,
                num_epochs=args.epochs, patience=args.patience,
                key=key, verbose=verbose,
            )
            dt = time.time() - t0

            test_accs.append(test_acc)
            if not verbose:
                print(f"test={test_acc:.2%}  ({dt:.1f}s)")

        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        baseline = PUBLISHED_BASELINES.get(ds_name, 0)
        gap = mean_acc * 100 - baseline

        print(f"\n  {ds_name} Result: {mean_acc:.2%} ± {std_acc:.2%}")
        print(f"  Published HGNN:  {baseline:.2f}%")
        print(f"  Gap:             {gap:+.1f}pt")

        results[ds_name] = {"mean": mean_acc, "std": std_acc, "gap": gap}

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Summary: {args.conv} | hidden={args.hidden_dim} | "
              f"residual={args.residual} | ln={args.layer_norm}")
        print(f"{'='*60}")
        print(f"  {'Dataset':<12} {'hgx':>8} {'Published':>10} {'Gap':>8}")
        print(f"  {'-'*40}")
        for ds, r in results.items():
            bl = PUBLISHED_BASELINES.get(ds, 0)
            print(f"  {ds:<12} {r['mean']*100:>7.2f}% {bl:>9.2f}% {r['gap']:>+7.1f}pt")

    return results


def main():
    parser = argparse.ArgumentParser(description="Standard citation dataset benchmark")
    parser.add_argument("--dataset", default="Cora", help="Cora, CiteSeer, PubMed, or all")
    parser.add_argument("--conv", default="HGNNConv", choices=list(CONV_MAP.keys()))
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--initial-alpha", type=float, default=None)
    parser.add_argument("--sparse", action="store_true", help="Use sparse incidence (for Pubmed)")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
