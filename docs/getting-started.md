---
category: infrastructure
section: methodology
weight: 10
title: "Getting Started"
status: draft
tags: [installation, tutorial, quickstart, hgx]
---

# Getting Started

## Installation

```bash
pip install hgx
```

For development (with tests):

```bash
git clone https://github.com/m9h/hgx.git
cd hgx
uv venv && uv pip install -e ".[tests]"
uv run pytest
```

## Quick start

### Create a hypergraph

```python
import jax
import jax.numpy as jnp
import hgx

# From an edge list: 4 nodes, 2 hyperedges
# Hyperedge 0 = {0, 1, 2}, Hyperedge 1 = {1, 2, 3}
hg = hgx.from_edge_list(
    [(0, 1, 2), (1, 2, 3)],
    node_features=jnp.ones((4, 16)),
)

# Or from an incidence matrix
H = jnp.array([[1, 0], [1, 1], [1, 1], [0, 1]], dtype=jnp.float32)
hg = hgx.from_incidence(H, node_features=jnp.ones((4, 16)))
```

### Convolution layers

hgx provides six convolution layers, all sharing the same interface:

```python
key = jax.random.PRNGKey(0)

# UniGCN — first-order message passing (mean aggregation)
conv = hgx.UniGCNConv(in_dim=16, out_dim=32, key=key)
out = conv(hg)  # (4, 32)

# UniGAT — attention-weighted aggregation
conv = hgx.UniGATConv(in_dim=16, out_dim=32, key=key)
out = conv(hg)  # (4, 32)

# UniGIN — GIN-style with learnable self-loop and MLP
conv = hgx.UniGINConv(in_dim=16, out_dim=32, key=key)
out = conv(hg)  # (4, 32)

# THNN — tensorized higher-order interactions via CP decomposition
conv = hgx.THNNConv(in_dim=16, out_dim=32, rank=64, key=key)
out = conv(hg)  # (4, 32)

# Sparse variants — O(nnz) instead of O(n*m), same results
conv = hgx.UniGCNSparseConv(in_dim=16, out_dim=32, key=key)
out = conv(hg)  # (4, 32)

conv = hgx.THNNSparseConv(in_dim=16, out_dim=32, rank=64, key=key)
out = conv(hg)  # (4, 32)
```

### Multi-layer model

```python
model = hgx.HGNNStack(
    conv_dims=[(16, 32), (32, 32)],
    conv_cls=hgx.UniGCNConv,
    readout_dim=2,
    dropout_rate=0.1,
    key=key,
)
out = model(hg, key=key, inference=False)  # (4, 2)
```

### Training with Optax

```python
import equinox as eqx
import optax

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def step(model, hg, labels, opt_state):
    def loss_fn(m):
        logits = m(hg, inference=True)
        return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(labels, 2), axis=-1))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state_new, loss
```

### Dynamic topology

```python
# Pre-allocate capacity for JIT-compatible topology changes
hg = hgx.preallocate(hg, max_nodes=10, max_edges=5)

# Add a node with features, connected to hyperedge 0
hg = hgx.add_node(
    hg,
    features=jnp.zeros(16),
    hyperedges=jnp.array([True, False, False, False, False]),
)

# Remove a node
hg = hgx.remove_node(hg, idx=0)

# Add a hyperedge connecting nodes 1 and 3
members = jnp.zeros(10, dtype=bool).at[1].set(True).at[3].set(True)
hg = hgx.add_hyperedge(hg, members=members)
```

### Continuous-time dynamics

Evolve node features as a Neural ODE or Neural SDE (requires `pip install hgx[dynamics]`):

```python
# Neural ODE: dx/dt = tanh(conv(Hypergraph(x(t), H)))
hg = hgx.from_edge_list(
    [(0, 1, 2), (1, 2, 3)],
    node_features=jnp.ones((4, 16)),
)
conv = hgx.UniGCNConv(in_dim=16, out_dim=16, key=key)
ode = hgx.HypergraphNeuralODE(conv)
hg_evolved = hgx.evolve(ode, hg, t0=0.0, t1=1.0)

# Neural SDE: dx = tanh(conv(...)) dt + sigma dW
sde = hgx.HypergraphNeuralSDE(
    conv, num_nodes=4, node_dim=16, key=key,
)
hg_evolved = hgx.evolve(sde, hg, t0=0.0, t1=1.0, key=key)
```

### Transforms

```python
# Clique expansion: hypergraph -> graph adjacency
A = hgx.clique_expansion(hg)

# Hypergraph Laplacian (normalized or unnormalized)
L = hgx.hypergraph_laplacian(hg, normalized=True)
```
