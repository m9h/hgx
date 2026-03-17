#!/usr/bin/env python3
"""Systematic recipe × model × dataset sweep for HGNN accuracy alignment.

Sweeps over published training recipes (HGNN, AllSet, custom) combined
with architectural variants (HGNNConv, UniGCNConv, ±residual, ±LN) on
Cora/Citeseer/Pubmed, running multiple seeds per configuration.

Usage:
    python scripts/recipe_alignment.py --dataset Cora --num-seeds 5
    python scripts/recipe_alignment.py --dataset all --num-seeds 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Import the benchmark runner
from benchmark_standard import (
    DATASETS,
    PUBLISHED_BASELINES,
    build_model,
    load_planetoid,
    train_and_eval,
)

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Recipe definitions (from published papers)
# ---------------------------------------------------------------------------

RECIPES = {
    "hgnn_original": {
        "description": "HGNN (Feng 2019) original recipe",
        "conv": "HGNNConv",
        "hidden_dim": 16,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "epochs": 200,
        "residual": False,
        "layer_norm": False,
        "initial_alpha": None,
    },
    "hgnn_residual": {
        "description": "HGNNConv + residual + LayerNorm",
        "conv": "HGNNConv",
        "hidden_dim": 16,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "epochs": 200,
        "residual": True,
        "layer_norm": True,
        "initial_alpha": None,
    },
    "hgnn_wide": {
        "description": "HGNNConv with hidden=64",
        "conv": "HGNNConv",
        "hidden_dim": 64,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "epochs": 200,
        "residual": False,
        "layer_norm": False,
        "initial_alpha": None,
    },
    "allset_recipe": {
        "description": "AllSet-style recipe (hidden=64, no WD)",
        "conv": "HGNNConv",
        "hidden_dim": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "epochs": 500,
        "residual": False,
        "layer_norm": False,
        "initial_alpha": None,
    },
    "unigcn_baseline": {
        "description": "UniGCNConv baseline (project-before)",
        "conv": "UniGCNConv",
        "hidden_dim": 16,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "epochs": 200,
        "residual": False,
        "layer_norm": False,
        "initial_alpha": None,
    },
    "hgnn_initial_residual": {
        "description": "HGNNConv + initial residual (alpha=0.1)",
        "conv": "HGNNConv",
        "hidden_dim": 16,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "epochs": 200,
        "residual": False,
        "layer_norm": True,
        "initial_alpha": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_recipe(recipe_name, recipe, ds_name, features, labels,
               train_mask, val_mask, test_mask, edge_list, num_seeds):
    """Run one recipe on one dataset for num_seeds seeds."""
    import hgx

    n, d = features.shape
    num_classes = int(labels.max()) + 1

    hg = hgx.from_edge_list(edge_list, num_nodes=n, node_features=features)

    test_accs = []
    for seed in range(num_seeds):
        key = jax.random.PRNGKey(seed)
        model = build_model(
            conv_name=recipe["conv"],
            in_dim=d,
            hidden_dim=recipe["hidden_dim"],
            num_classes=num_classes,
            dropout=recipe["dropout"],
            residual=recipe["residual"],
            layer_norm=recipe["layer_norm"],
            initial_alpha=recipe["initial_alpha"],
            key=key,
        )

        test_acc, _ = train_and_eval(
            model, hg, labels, train_mask, val_mask, test_mask,
            lr=recipe["lr"],
            weight_decay=recipe["weight_decay"],
            num_epochs=recipe["epochs"],
            key=key,
            verbose=False,
        )
        test_accs.append(test_acc)

    return {
        "mean": float(np.mean(test_accs)),
        "std": float(np.std(test_accs)),
        "accs": [float(a) for a in test_accs],
    }


def main():
    parser = argparse.ArgumentParser(description="Recipe alignment sweep")
    parser.add_argument("--dataset", default="Cora", help="Cora, CiteSeer, PubMed, or all")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--recipes", nargs="+", default=None,
                        help="Subset of recipes to run (default: all)")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset.lower() == "all" else [args.dataset]
    recipe_names = args.recipes or list(RECIPES.keys())

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Loading {ds_name}...")
        print(f"{'='*70}")

        features, labels, train_mask, val_mask, test_mask, edge_list = load_planetoid(ds_name)
        n, d = features.shape
        print(f"  Nodes={n}  Features={d}  Hyperedges={len(edge_list)}")

        ds_results = {}
        for rname in recipe_names:
            recipe = RECIPES[rname]
            print(f"\n  [{rname}] {recipe['description']}")
            print(f"    conv={recipe['conv']} hidden={recipe['hidden_dim']} "
                  f"lr={recipe['lr']} wd={recipe['weight_decay']} "
                  f"dropout={recipe['dropout']} res={recipe['residual']} "
                  f"ln={recipe['layer_norm']} alpha={recipe['initial_alpha']}")

            t0 = time.time()
            result = run_recipe(
                rname, recipe, ds_name,
                features, labels, train_mask, val_mask, test_mask,
                edge_list, args.num_seeds,
            )
            dt = time.time() - t0

            baseline = PUBLISHED_BASELINES.get(ds_name, 0)
            gap = result["mean"] * 100 - baseline
            print(f"    => {result['mean']*100:.2f}% ± {result['std']*100:.2f}%  "
                  f"(gap: {gap:+.1f}pt)  [{dt:.1f}s]")

            ds_results[rname] = result

        all_results[ds_name] = ds_results

    # Print summary table
    print(f"\n{'='*70}")
    print("  RECIPE ALIGNMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Recipe':<25} ", end="")
    for ds in datasets:
        print(f"{'  ' + ds:>12}", end="")
    print()
    print(f"  {'-'*25} ", end="")
    for _ in datasets:
        print(f"  {'----------':>12}", end="")
    print()

    for rname in recipe_names:
        print(f"  {rname:<25} ", end="")
        for ds in datasets:
            r = all_results[ds][rname]
            print(f"  {r['mean']*100:>6.2f}±{r['std']*100:.1f}", end="")
        print()

    print(f"\n  {'Published HGNN':<25} ", end="")
    for ds in datasets:
        bl = PUBLISHED_BASELINES.get(ds, 0)
        print(f"  {bl:>10.2f}", end="")
    print()

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "num_seeds": args.num_seeds,
                "results": all_results,
                "baselines": {ds: PUBLISHED_BASELINES.get(ds, 0) for ds in datasets},
            }, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
