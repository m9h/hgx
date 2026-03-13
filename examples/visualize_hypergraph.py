"""Visualize a small hypergraph using hgx drawing utilities.

Creates a 6-node, 4-hyperedge example and saves plots to
``examples/hypergraph_viz.png``.

Usage::

    uv run python examples/visualize_hypergraph.py
"""

from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import hgx
import jax.numpy as jnp
import matplotlib.pyplot as plt


# ---------- Build a small hypergraph ----------
# 6 nodes, 4 hyperedges of varying size:
#   e0 = {0, 1, 2}        (triangle)
#   e1 = {2, 3}            (pair)
#   e2 = {3, 4, 5}         (triangle)
#   e3 = {0, 1, 3, 4, 5}   (large hyperedge)
hg = hgx.from_edge_list(
    [(0, 1, 2), (2, 3), (3, 4, 5), (0, 1, 3, 4, 5)],
    num_nodes=6,
    node_features=jnp.eye(6),
)

# ---------- Draw ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

hgx.draw_hypergraph(
    hg,
    ax=axes[0],
    title="Hypergraph (star expansion layout)",
)

hgx.draw_incidence(
    hg,
    ax=axes[1],
    title="Incidence matrix",
)

fig.tight_layout()
out_path = Path(__file__).parent / "hypergraph_viz.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved to {out_path}")
