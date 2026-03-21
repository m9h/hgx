---
category: research
section: introduction
weight: 10
title: "HGX: Hypergraph Neural Networks in JAX/Equinox"
status: draft
slide_summary: "HGX is the first JAX-native library for deep learning on hypergraphs and higher-order topological domains, providing 14 convolution layers, continuous-time dynamics via Diffrax, and JIT-compatible dynamic topology -- all built on Equinox."
tags: [hgx, hypergraph, jax, equinox, topological-deep-learning, higher-order-networks]
---

# hgx

Hypergraph neural networks in JAX/Equinox.

**hgx** provides the first JAX-native library for deep learning on hypergraphs and higher-order topological domains, built on [Equinox](https://docs.kidger.site/equinox/) and designed to compose with the broader Kidger stack ([Diffrax](https://docs.kidger.site/diffrax/), [Optax](https://github.com/google-deepmind/optax), etc.).

## Why hypergraphs?

Standard graphs model pairwise relationships. But many systems — cell signaling networks, protein complexes, co-authorship, chemical reactions — involve **multi-way interactions** that pairwise edges cannot capture. A hypergraph generalizes a graph by allowing each edge (hyperedge) to connect any number of vertices simultaneously.

## Features

- **Core data structure** (`Hypergraph`) with incidence matrix representation, optional geometry (Euclidean, Poincare, Lorentz), and masking for dynamic topology
- **14 convolution layers:**
  - `UniGCNConv` — first-order sum-aggregation message passing; reduces to GCN on pairwise graphs
  - `UniGCNSparseConv` — segment-sum drop-in replacement for UniGCN (O(nnz) instead of O(n*m))
  - `UniGATConv` — learned attention weights in the hyperedge-to-vertex step
  - `UniGINConv` — GIN-style MLP aggregation with a learnable self-loop parameter
  - `THNNConv` — tensorized high-order interactions via CP decomposition ([Wang et al., SDM 2024](https://arxiv.org/abs/2306.02560))
  - `THNNSparseConv` — sparse variant of THNN
  - `SheafHypergraphConv` — sheaf-theoretic message passing
  - `SheafDiffusion` — diffusion on sheaves
  - `PoincareHypergraphConv` — Poincaré ball hyperbolic geometry (requires `hgx[geometry]`)
  - `LorentzHypergraphConv` — Lorentz hyperboloid geometry (requires `hgx[geometry]`)
  - `ProductSpaceConv`, `ProductManifoldConv` — mixed-curvature product manifolds
  - `SE3HypergraphConv` — SE(3)-equivariant layers (requires `hgx[geometry]`)
  - `OTConv` — optimal transport barycenter aggregation
- **HGNNStack** — multi-layer model builder with activation, dropout, and optional readout
- **Dynamic topology** — `preallocate`, `add_node`, `remove_node`, `add_hyperedge`, `remove_hyperedge` (all JIT-compatible)
- **Sparse message passing** — `incidence_to_star_expansion`, `vertex_to_edge`, `edge_to_vertex` via `jax.ops.segment_sum`
- **Continuous dynamics** — `HypergraphNeuralODE`, `HypergraphNeuralSDE`, `HypergraphNeuralCDE` via [Diffrax](https://docs.kidger.site/diffrax/), with `LatentHypergraphODE/SDE` and Riemannian dynamics (requires `hgx[dynamics]`)
- **Information geometry** — Fisher-Rao metric, natural gradient descent, free-energy drift on the simplex
- **Optimal transport** — Sinkhorn, Wasserstein distance/barycenters, Gromov-Wasserstein, hypergraph alignment
- **Spectral methods** — hypergraph wavelet transforms, scattering, Chebyshev filters, Cheeger bounds
- **Topology** — persistent homology, Hodge Laplacians, topological features (requires `hgx[topology]`)
- **Temporal hypergraphs** — snapshot sequences, topology alignment, temporal smoothness loss
- **NDP** — Neural Developmental Programs with cell growth and division dynamics
- **Perturbation prediction** — in silico knockout screens, perturbation encoders
- **GRN loaders** — load gene regulatory networks from CSV, edge lists, or AnnData
- **PGMax bridge** — convert hypergraphs to factor graphs for probabilistic inference (requires `hgx[pgmax]`)
- **Visualization** — `draw_hypergraph`, `draw_incidence`, `draw_attention`, `draw_trajectory`, `draw_phase_portrait` (requires `hgx[viz]`)
- **Transforms** — clique expansion, hypergraph Laplacian
- **JAX-native** — JIT, vmap, and grad all work out of the box
- **Equinox modules** — composable with any Equinox/JAX workflow

## Installation

```bash
pip install hgx                            # core library
pip install "hgx[dynamics]"                # adds diffrax for Neural ODE/SDE/CDE
pip install "hgx[geometry]"                # adds e3nn-jax for SE(3) layers
pip install "hgx[topology]"               # adds giotto-tda for persistent homology
pip install "hgx[viz]"                     # adds matplotlib + networkx
pip install "hgx[dynamics,geometry,viz]"   # everything
```

With conda (once the feedstock is published):

```bash
conda install -c conda-forge hgx
```

For development:

```bash
git clone https://github.com/m9h/hgx.git
cd hgx
uv venv && uv pip install -e ".[tests,dynamics,viz]"
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

## Continuous dynamics

Requires the `dynamics` extra (`pip install "hgx[dynamics]"`). Node features evolve as a Neural ODE or Neural SDE whose vector field is a hypergraph convolution:

```python
import jax
import jax.numpy as jnp
import hgx

hg = hgx.from_edge_list(
    [(0, 1, 2), (1, 2, 3)],
    node_features=jnp.ones((4, 8)),
)

# Build a Neural ODE: dx/dt = tanh(conv(x(t), H))
conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=jax.random.PRNGKey(0))
neural_ode = hgx.HypergraphNeuralODE(conv)

# Integrate from t=0 to t=1
sol = neural_ode(hg, t0=0.0, t1=1.0, dt0=0.1)
final_features = sol.ys[-1]  # (4, 8)

# Or use the convenience wrapper
hg_evolved = hgx.evolve(neural_ode, hg, t0=0.0, t1=1.0)

# Neural SDE for stochastic dynamics
sde = hgx.HypergraphNeuralSDE(
    conv, num_nodes=4, node_dim=8,
    sigma_init=0.1, key=jax.random.PRNGKey(1),
)
hg_stochastic = hgx.evolve(sde, hg, t0=0.0, t1=1.0, key=jax.random.PRNGKey(2))
```

See [`examples/neural_ode.py`](examples/neural_ode.py) for a complete example with trajectory visualization.

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
- **Geometric embeddings** (Euclidean, Poincare, and Lorentz positions via the `geometry` field)
- **Dynamic topology** (node/edge birth via pre-allocated masked arrays)
- **Continuous-time evolution** (Neural ODE/SDE on hypergraphs via Diffrax)

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
| Geometry field (Euclidean, Poincare, Lorentz) | Done |
| Diffrax integration (Neural ODE/SDE/CDE) | Done |
| Sheaf convolutions (SheafHypergraphConv, SheafDiffusion) | Done |
| Hyperbolic convolutions (Poincare, Lorentz, product manifolds) | Done |
| SE(3)-equivariant hypergraph layers | Done |
| Information geometry (Fisher-Rao, natural gradients) | Done |
| Optimal transport (Sinkhorn, Wasserstein, Gromov-Wasserstein) | Done |
| Spectral methods (wavelets, scattering, Chebyshev) | Done |
| Persistent homology and Hodge Laplacians | Done |
| Temporal hypergraphs (snapshots, topology alignment) | Done |
| NDP (Neural Developmental Programs) on hypergraphs | Done |
| Perturbation prediction and knockout screens | Done |
| GRN loaders (CSV, edge list, AnnData) | Done |
| PGMax bridge (factor graphs, active inference) | Done |
| Pooling (TopK, Spectral, Hierarchical) | Done |
| CI + docs | In progress |

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
