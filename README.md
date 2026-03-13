# hgx

Hypergraph neural networks in JAX/Equinox.

**hgx** provides the first JAX-native library for deep learning on hypergraphs and higher-order topological domains, built on [Equinox](https://docs.kidger.site/equinox/) and designed to compose with the broader Kidger stack ([Diffrax](https://docs.kidger.site/diffrax/), [Optax](https://github.com/google-deepmind/optax), etc.).

## Why hypergraphs?

Standard graphs model pairwise relationships. But many systems — cell signaling networks, protein complexes, co-authorship, chemical reactions — involve **multi-way interactions** that pairwise edges cannot capture. A hypergraph generalizes a graph by allowing each edge (hyperedge) to connect any number of vertices simultaneously.

## Features

- **Core data structure** (`Hypergraph`) with incidence matrix representation, optional geometry (3D positions), and masking for dynamic topology
- **First-order convolution** (`UniGCNConv`) — two-stage vertex-to-hyperedge-to-vertex message passing. Reduces to GCN on pairwise graphs.
- **Tensorized convolution** (`THNNConv`) — high-order multilinear interactions via CP decomposition. Captures joint effects that sum-aggregation misses. First open-source implementation of [Wang et al., SDM 2024](https://arxiv.org/abs/2306.02560).
- **Transforms** — clique expansion, hypergraph Laplacian
- **JAX-native** — JIT, vmap, and grad all work out of the box
- **Equinox modules** — composable with any Equinox/JAX workflow

## Installation

```bash
pip install hgx
```

Or for development:

```bash
git clone https://github.com/m9h/hgx.git
cd hgx
uv venv && uv pip install -e ".[tests]"
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

# Tensorized convolution (THNN) — captures higher-order interactions
conv_ho = hgx.THNNConv(in_dim=16, out_dim=32, rank=64, key=jax.random.PRNGKey(1))
out_ho = conv_ho(hg)  # (4, 32)

# Gradients work
def loss_fn(model):
    return jnp.sum(model(hg))
grads = jax.grad(loss_fn)(conv)
```

## Design

The data structure is designed to be forward-compatible with:
- **Combinatorial complexes** (multi-rank cells with hierarchy)
- **Geometric embeddings** (3D positions, SE(3) equivariance)
- **Dynamic topology** (node/edge birth via pre-allocated masked arrays)
- **Diffrax integration** (neural SDEs/ODEs on evolving hypergraphs)

while keeping the common hypergraph case simple.

## Roadmap

| Phase | Status |
|-------|--------|
| Static hypergraph convolutions (UniGCN, THNN) | Done |
| Clique expansion, Laplacian | Done |
| JIT/grad/vmap compatibility | Done |
| Dynamic topology (node birth/death) | Planned |
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
