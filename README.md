# hgx

Hypergraph neural networks in JAX/Equinox.

**hgx** provides the first JAX-native library for deep learning on hypergraphs and higher-order topological domains, built on [Equinox](https://docs.kidger.site/equinox/) and designed to compose with the broader Kidger stack ([Diffrax](https://docs.kidger.site/diffrax/), [Optax](https://github.com/google-deepmind/optax), etc.).

## Why hypergraphs?

Standard graphs model pairwise relationships. But many systems — cell signaling networks, protein complexes, co-authorship, chemical reactions — involve **multi-way interactions** that pairwise edges cannot capture. A hypergraph generalizes a graph by allowing each edge (hyperedge) to connect any number of vertices simultaneously.

## Features

- **Core data structure** (`Hypergraph`) with incidence matrix representation, optional geometry (Euclidean, Poincaré, Lorentz), and masking for dynamic topology
- **6 convolution layers:**
  - `UniGCNConv` — first-order sum-aggregation message passing; reduces to GCN on pairwise graphs
  - `UniGCNSparseConv` — segment-sum drop-in replacement for UniGCN (O(nnz) instead of O(n·m))
  - `UniGATConv` — learned attention weights in the hyperedge → vertex step
  - `UniGINConv` — GIN-style MLP aggregation with a learnable self-loop parameter ε
  - `THNNConv` — tensorized high-order interactions via CP decomposition ([Wang et al., SDM 2024](https://arxiv.org/abs/2306.02560))
  - `THNNSparseConv` — sparse variant of THNN
- **HGNNStack** — multi-layer model builder with activation, dropout, and optional readout
- **Dynamic topology** — `preallocate`, `add_node`, `remove_node`, `add_hyperedge`, `remove_hyperedge` (all JIT-compatible)
- **Sparse message passing** — `incidence_to_star_expansion`, `vertex_to_edge`, `edge_to_vertex` via `jax.ops.segment_sum`
- **Visualization** — `draw_hypergraph`, `draw_incidence`, `draw_attention` (requires `hgx[viz]`)
- **Transforms** — clique expansion, hypergraph Laplacian
- **JAX-native** — JIT, vmap, and grad all work out of the box
- **Equinox modules** — composable with any Equinox/JAX workflow

## Installation

```bash
pip install hgx
```

With visualization support:

```bash
pip install "hgx[viz]"
```

For development:

```bash
git clone https://github.com/m9h/hgx.git
cd hgx
uv venv && uv pip install -e ".[tests,viz]"
uv run pytest
```

## Quick start

```python
import jax
import jax.numpy as jnp
import hgx

# Create a hypergraph: 4 nodes, 2 hyperedges
# Hyperedge 0 = {0, 1, 2}, Hyperedge 1 = {1, 2, 3}
hg = hgx.from_edge_list(
    [(0, 1, 2), (1, 2, 3)],
    node_features=jnp.ones((4, 16)),
)

# First-order convolution (UniGCN)
conv = hgx.UniGCNConv(in_dim=16, out_dim=32, key=jax.random.PRNGKey(0))
out = conv(hg)  # (4, 32)

# Attention-based convolution (UniGAT)
attn_conv = hgx.UniGATConv(in_dim=16, out_dim=32, key=jax.random.PRNGKey(1))
out_attn = attn_conv(hg)  # (4, 32)

# Tensorized convolution (THNN) — captures higher-order interactions
conv_ho = hgx.THNNConv(in_dim=16, out_dim=32, rank=64, key=jax.random.PRNGKey(2))
out_ho = conv_ho(hg)  # (4, 32)

# Multi-layer model with HGNNStack
model = hgx.HGNNStack(
    conv_dims=[(16, 32), (32, 32)],
    conv_cls=hgx.UniGCNConv,
    readout_dim=4,
    dropout_rate=0.1,
    key=jax.random.PRNGKey(3),
)
logits = model(hg, key=jax.random.PRNGKey(4))  # (4, 4)

# Gradients work
def loss_fn(m):
    return jnp.sum(m(hg, inference=True))
grads = jax.grad(loss_fn)(model)
```

## Dynamic topology

Grow or shrink a hypergraph at runtime — all operations are JIT-compatible:

```python
# Pre-allocate capacity for up to 8 nodes and 4 hyperedges
hg = hgx.preallocate(hg, max_nodes=8, max_edges=4)

# Add a new node with features, connected to hyperedge 0
new_feats = jnp.ones(16)
membership = jnp.array([True, False])  # belongs to edge 0 only
hg = hgx.add_node(hg, features=new_feats, hyperedges=membership)

# Add a new hyperedge spanning nodes 0 and 3
members = jnp.array([True, False, False, True, False, False, False, False])
hg = hgx.add_hyperedge(hg, members=members)

# Convolutions work on the updated topology
out = conv(hg)
```

## Visualization

Requires the `viz` extra (`pip install "hgx[viz]"`):

```python
import hgx

hg = hgx.from_edge_list([(0, 1, 2), (2, 3), (3, 4, 5)])

# Draw bipartite star-expansion layout
ax = hgx.draw_hypergraph(hg, title="My hypergraph")

# Show the incidence matrix as a heatmap
ax = hgx.draw_incidence(hg)
```

See [`examples/visualize_hypergraph.py`](examples/visualize_hypergraph.py) for a complete example.

## Design

The data structure is designed to be forward-compatible with:
- **Combinatorial complexes** (multi-rank cells with hierarchy)
- **Geometric embeddings** (Euclidean, Poincaré, and Lorentz positions via the `geometry` field)
- **Dynamic topology** (node/edge birth via pre-allocated masked arrays)
- **Diffrax integration** (neural SDEs/ODEs on evolving hypergraphs)

while keeping the common hypergraph case simple.

## Roadmap

| Feature | Status |
|---------|--------|
| Static hypergraph convolutions (UniGCN, THNN) | Done |
| Clique expansion, Laplacian | Done |
| JIT/grad/vmap compatibility | Done |
| Attention convolution (UniGAT) | Done |
| GIN convolution (UniGIN) | Done |
| Sparse variants (UniGCNSparse, THNNSparse) | Done |
| Dynamic topology (add/remove nodes & edges) | Done |
| HGNNStack multi-layer model builder | Done |
| Visualization (draw_hypergraph, draw_incidence, draw_attention) | Done |
| Geometry field (Euclidean, Poincaré, Lorentz) | Done |
| CI + docs | In progress |
| Diffrax integration (Neural SDE on growth) | Planned |
| NDP (Neural Developmental Programs) on hypergraphs | Planned |
| SE(3)-equivariant hypergraph layers | Planned |

## Related work

- [UniGNN](https://arxiv.org/abs/2105.00956) (IJCAI 2021) — unified GNN framework for hypergraphs
- [THNN](https://arxiv.org/abs/2306.02560) (SDM 2024) — tensorized hypergraph neural networks
- [DHG](https://github.com/iMoonLab/DeepHypergraph) — PyTorch hypergraph library
- [TopoModelX](https://github.com/pyt-team/TopoModelX) — PyTorch topological deep learning
- [DevoGraph](https://github.com/DevoLearn/DevoGraph) — GNNs for C. elegans developmental biology

## Context

This library was initiated as part of research toward [GSoC 2026 DevoGraph](https://neurostars.org/t/gsoc-2026-project-6-openworm-devoworm-devograph/35565) (OpenWorm/DevoWorm), with the goal of providing JAX-native tools for modeling developmental biology as evolving hypergraph dynamics.

## License

Apache 2.0
