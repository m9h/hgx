---
category: infrastructure
section: introduction
weight: 20
title: "Documentation Index"
status: draft
tags: [documentation, hgx, overview]
---

# hgx

**Hypergraph neural networks in JAX/Equinox.**

hgx provides a JAX-native library for deep learning on hypergraphs and higher-order topological domains, built on [Equinox](https://docs.kidger.site/equinox/) and designed to compose with the broader JAX ecosystem ([Optax](https://github.com/google-deepmind/optax), [Diffrax](https://docs.kidger.site/diffrax/), etc.).

## Why hypergraphs?

Standard graphs model pairwise relationships. But many systems — cell signaling networks, protein complexes, co-authorship, chemical reactions — involve **multi-way interactions** that pairwise edges cannot capture. A hypergraph generalizes a graph by allowing each edge (hyperedge) to connect any number of vertices simultaneously.

## Features

- **Core data structure** (`Hypergraph`) with incidence matrix representation, optional geometry (3D positions), and masking for dynamic topology
- **Six convolution layers** — `UniGCNConv`, `UniGATConv`, `UniGINConv`, `THNNConv`, plus sparse variants `UniGCNSparseConv` and `THNNSparseConv`
- **Composable model** (`HGNNStack`) for multi-layer architectures with dropout and readout
- **Dynamic topology** — `preallocate`, `add_node`, `add_hyperedge`, `remove_node`, `remove_hyperedge` for JIT-compatible topology changes
- **Transforms** — clique expansion, hypergraph Laplacian
- **Continuous-time dynamics** — `HypergraphNeuralODE` and `HypergraphNeuralSDE` via [Diffrax](https://docs.kidger.site/diffrax/) for evolving node features as differential equations
- **Sparse utilities** — index-based O(nnz) message passing via star expansion
- **Visualization** — hypergraph drawing, incidence heatmaps, attention weight visualization
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

## Design

The data structure is designed to be forward-compatible with:

- **Combinatorial complexes** (multi-rank cells with hierarchy)
- **Geometric embeddings** (3D positions, SE(3) equivariance)
- **Dynamic topology** (node/edge birth via pre-allocated masked arrays)
- **Diffrax integration** (neural SDEs/ODEs on evolving hypergraphs)

## Related work

- [UniGNN](https://arxiv.org/abs/2105.00956) (IJCAI 2021) — unified GNN framework for hypergraphs
- [THNN](https://arxiv.org/abs/2306.02560) (SDM 2024) — tensorized hypergraph neural networks
- [DHG](https://github.com/iMoonLab/DeepHypergraph) — PyTorch hypergraph library
- [TopoModelX](https://github.com/pyt-team/TopoModelX) — PyTorch topological deep learning

## License

Apache 2.0
