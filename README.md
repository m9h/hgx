# hgx

**Hypergraph neural networks in JAX/Equinox.**

**hgx** is a JAX-native library for deep learning on hypergraphs and higher-order topological domains, providing a foundation for modeling complex, multi-way interactions.

📚 [Documentation](https://m9h.github.io/hgx) | 🐛 [Source Code](https://github.com/m9h/hgx)

## Features

- **6 Convolution Layers**: A suite of standard and sparse hypergraph message-passing layers.
- **Dynamic Topology**: JIT-compatible pre-allocation, node/edge additions, and removals.
- **Continuous Dynamics**: Hypergraph Neural ODEs, SDEs, and CDEs via Diffrax.
- **Neural Developmental Programs (NDP)**: Cell lineage and fate modeling on hypergraphs.
- **Visualization**: Built-in plotting for hypergraphs, incidence matrices, and trajectories.
- ***C. elegans* Data Loaders**: Ready-to-use biological datasets (connectome, cell lineage, DevoGraph).

## Installation

```bash
# Core library
pip install hgx

# With optional dependencies
pip install "hgx[viz]"       # Visualization
pip install "hgx[dynamics]"  # Continuous dynamics (Diffrax)
pip install "hgx[pgmax]"     # Factor graph inference
```

Or via Conda:
```bash
conda install -c conda-forge hgx
```

## Quickstart

Build a hypergraph and apply a convolution layer in just a few lines:

```python
import jax, jax.numpy as jnp, hgx

# Create a hypergraph from an incidence matrix (4 nodes, 2 edges)
H = jnp.array([[1, 0], [1, 1], [0, 1], [0, 1]], dtype=jnp.float32)
hg = hgx.from_incidence(H, node_features=jnp.ones((4, 16)))

# Apply a first-order hypergraph convolution
conv = hgx.UniGCNConv(in_dim=16, out_dim=32, key=jax.random.PRNGKey(0))
out = conv(hg)  # shape: (4, 32)
```
*See `examples/synthetic_classification.py` for a full training loop.*

## Continuous Dynamics

Model evolving systems with Neural ODEs, SDEs, or CDEs on hypergraphs:

```python
import jax, jax.numpy as jnp, hgx

hg = hgx.from_incidence(jnp.array([[1, 1], [1, 0], [0, 1]], dtype=jnp.float32), node_features=jnp.ones((3, 8)))
conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=jax.random.PRNGKey(0))

# Build and integrate a Neural ODE: dx/dt = conv(x(t), H)
neural_ode = hgx.HypergraphNeuralODE(conv)
sol = neural_ode(hg, t0=0.0, t1=1.0, dt0=0.1)

final_features = sol.ys[-1]  # shape: (3, 8)
```
*See `examples/neural_ode.py` and `examples/neural_ode_training.py` for more details.*

## Convolution Layers

| Layer | Description |
|-------|-------------|
| `UniGCNConv` | First-order sum-aggregation message passing. |
| `UniGCNSparseConv` | Sparse `O(nnz)` drop-in replacement for `UniGCNConv`. |
| `UniGATConv` | Learned attention weights for hyperedge-to-vertex messages. |
| `UniGINConv` | GIN-style MLP aggregation with learnable self-loops. |
| `THNNConv` | Tensorized high-order interactions via CP decomposition. |
| `THNNSparseConv` | Sparse variant of `THNNConv` for larger hypergraphs. |

## Context

This library was initiated as part of research toward [GSoC 2026 DevoGraph](https://neurostars.org/t/gsoc-2026-project-6-openworm-devoworm-devograph/35565) (OpenWorm/DevoWorm), with the goal of providing JAX-native tools for modeling developmental biology as evolving hypergraph dynamics.

## License & Citation

**hgx** is licensed under the Apache License 2.0. If you use this library in your research, please link back to the repository.
