#!/usr/bin/env python3
"""Speed comparison: hgx (JAX) vs DHG (PyTorch) on standard datasets.

Measures forward pass (inference) and training step timing for both
frameworks on Cora, Citeseer, and Pubmed.

Usage:
    python scripts/speed_comparison.py --dataset Cora
    python scripts/speed_comparison.py --dataset all --num-warmup 5 --num-runs 50
"""

from __future__ import annotations

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import hgx
from benchmark_standard import load_planetoid, DATASETS


# ---------------------------------------------------------------------------
# hgx timing
# ---------------------------------------------------------------------------

def time_hgx(features, labels, train_mask, edge_list, *,
             hidden_dim=16, num_warmup=5, num_runs=50):
    """Time hgx forward and training step."""
    n, d = features.shape
    num_classes = int(labels.max()) + 1

    hg = hgx.from_edge_list(edge_list, num_nodes=n, node_features=features)
    key = jax.random.PRNGKey(0)

    model = hgx.HGNNStack(
        conv_dims=[(d, hidden_dim), (hidden_dim, hidden_dim)],
        conv_cls=hgx.HGNNConv,
        readout_dim=num_classes,
        dropout_rate=0.5,
        key=key,
    )

    # JIT-compile
    @eqx.filter_jit
    def forward(model, hg):
        return model(hg, inference=True)

    @eqx.filter_jit
    def train_step(model, hg, labels, train_mask, opt_state, key):
        def loss_fn(model):
            logits = model(hg, key=key, inference=False)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            one_hot = jax.nn.one_hot(labels, num_classes)
            per_node = -jnp.sum(one_hot * log_probs, axis=-1)
            return jnp.sum(per_node * train_mask) / jnp.sum(train_mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    optimizer = optax.adamw(0.01, weight_decay=5e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Warmup
    for _ in range(num_warmup):
        _ = forward(model, hg).block_until_ready()
        key, subkey = jax.random.split(key)
        model, opt_state, _ = train_step(model, hg, labels, train_mask, opt_state, subkey)

    # Time inference
    times_fwd = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = forward(model, hg).block_until_ready()
        times_fwd.append(time.perf_counter() - t0)

    # Time training step
    times_train = []
    for _ in range(num_runs):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        model, opt_state, _ = train_step(model, hg, labels, train_mask, opt_state, subkey)
        jax.block_until_ready((model, opt_state))
        times_train.append(time.perf_counter() - t0)

    return {
        "inference_ms": np.median(times_fwd) * 1000,
        "train_step_ms": np.median(times_train) * 1000,
        "inference_std_ms": np.std(times_fwd) * 1000,
        "train_step_std_ms": np.std(times_train) * 1000,
    }


# ---------------------------------------------------------------------------
# DHG timing (if available)
# ---------------------------------------------------------------------------

def time_dhg(features_np, labels_np, train_mask_np, edge_list, *,
             hidden_dim=16, num_warmup=5, num_runs=50):
    """Time DHG forward and training step."""
    try:
        import torch
        import dhg
        from dhg.models import HGNN as HGNN_DHG
    except ImportError:
        return None

    device = torch.device("cpu")
    n, d = features_np.shape
    num_classes = int(labels_np.max()) + 1

    X = torch.tensor(features_np, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)
    train_mask_t = torch.tensor(train_mask_np, dtype=torch.bool, device=device)

    hg = dhg.Hypergraph(n, edge_list)

    model = HGNN_DHG(d, hidden_dim, num_classes, use_bn=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(X, hg)
        model.train()
        optimizer.zero_grad()
        out = model(X, hg)
        loss = criterion(out[train_mask_t], labels_t[train_mask_t])
        loss.backward()
        optimizer.step()

    # Time inference
    model.eval()
    times_fwd = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(X, hg)
        times_fwd.append(time.perf_counter() - t0)

    # Time training step
    model.train()
    times_train = []
    for _ in range(num_runs):
        optimizer.zero_grad()
        t0 = time.perf_counter()
        out = model(X, hg)
        loss = criterion(out[train_mask_t], labels_t[train_mask_t])
        loss.backward()
        optimizer.step()
        times_train.append(time.perf_counter() - t0)

    return {
        "inference_ms": np.median(times_fwd) * 1000,
        "train_step_ms": np.median(times_train) * 1000,
        "inference_std_ms": np.std(times_fwd) * 1000,
        "train_step_std_ms": np.std(times_train) * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Speed comparison: hgx vs DHG")
    parser.add_argument("--dataset", default="Cora", help="Cora, CiteSeer, PubMed, or all")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=50)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset.lower() == "all" else [args.dataset]

    results = {}
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Speed comparison: {ds_name}")
        print(f"{'='*60}")

        features, labels, train_mask, val_mask, test_mask, edge_list = load_planetoid(ds_name)
        n, d = features.shape
        print(f"  Nodes: {n}  Features: {d}  Hyperedges: {len(edge_list)}")

        features_np = np.array(features)
        labels_np = np.array(labels)
        train_mask_np = np.array(train_mask)

        print(f"\n  Timing hgx (JAX)...")
        hgx_times = time_hgx(
            features, labels, train_mask, edge_list,
            hidden_dim=args.hidden_dim,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )
        print(f"    Inference: {hgx_times['inference_ms']:.2f} ms")
        print(f"    Train step: {hgx_times['train_step_ms']:.2f} ms")

        print(f"\n  Timing DHG (PyTorch)...")
        dhg_times = time_dhg(
            features_np, labels_np, train_mask_np, edge_list,
            hidden_dim=args.hidden_dim,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )

        if dhg_times:
            print(f"    Inference: {dhg_times['inference_ms']:.2f} ms")
            print(f"    Train step: {dhg_times['train_step_ms']:.2f} ms")

            speedup_inf = dhg_times["inference_ms"] / hgx_times["inference_ms"]
            speedup_train = dhg_times["train_step_ms"] / hgx_times["train_step_ms"]
            print(f"\n  Speedup (hgx vs DHG):")
            print(f"    Inference: {speedup_inf:.1f}x")
            print(f"    Training:  {speedup_train:.1f}x")
        else:
            print("    DHG not installed, skipping comparison")

        results[ds_name] = {"hgx": hgx_times, "dhg": dhg_times}

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  SPEED SUMMARY (median ms)")
        print(f"{'='*60}")
        print(f"  {'Dataset':<12} {'hgx fwd':>9} {'DHG fwd':>9} {'Speedup':>9}")
        print(f"  {'-'*42}")
        for ds, r in results.items():
            hgx_ms = r["hgx"]["inference_ms"]
            if r["dhg"]:
                dhg_ms = r["dhg"]["inference_ms"]
                speedup = dhg_ms / hgx_ms
                print(f"  {ds:<12} {hgx_ms:>8.2f} {dhg_ms:>8.2f} {speedup:>8.1f}x")
            else:
                print(f"  {ds:<12} {hgx_ms:>8.2f}     {'N/A':>5}     {'N/A':>5}")


if __name__ == "__main__":
    main()
