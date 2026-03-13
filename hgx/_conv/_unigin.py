"""UniGIN convolution layer for hypergraphs.

Implements the UniGIN variant from:
    Huang & Yang (2021). "UniGNN: a Unified Framework for Graph and
    Hypergraph Neural Networks." IJCAI 2021.

Key difference from UniGCN: uses a 2-layer MLP instead of a single linear
layer, and adds a learnable epsilon parameter for self-loop weighting:
    e_k = sum_{i in e_k} h_i
    h_i = MLP((1 + epsilon) * h_i + sum_{k: i in e_k} e_k)

When applied to a 2-uniform hypergraph, this reduces to standard GIN.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class UniGINConv(AbstractHypergraphConv):
    """UniGIN: Graph Isomorphism Network variant for hypergraphs.

    Performs two-stage message passing with a learnable self-loop weight:
        e_k = sum_{i in e_k} h_i
        h_i = MLP((1 + epsilon) * h_i + sum_{k: i in e_k} e_k)

    Attributes:
        mlp: 2-layer MLP (in_dim -> hidden_dim -> out_dim) with ReLU.
        epsilon: Learnable scalar weighting the self-loop.
    """

    mlp: eqx.nn.MLP
    epsilon: Float[Array, ""]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize UniGINConv.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            hidden_dim: Hidden dimension of the 2-layer MLP. Defaults to out_dim.
            key: PRNG key for weight initialization.
        """
        if hidden_dim is None:
            hidden_dim = out_dim
        self.mlp = eqx.nn.MLP(
            in_size=in_dim,
            out_size=out_dim,
            width_size=hidden_dim,
            depth=1,
            activation=jax.nn.relu,
            key=key,
        )
        self.epsilon = jnp.array(0.0)

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply UniGIN convolution.

        Args:
            hg: Input hypergraph.

        Returns:
            Updated node features of shape (num_nodes, out_dim).
        """
        H = hg._masked_incidence()
        x = hg.node_features

        # Stage 1: Vertex -> Hyperedge (sum aggregation, no normalization)
        e = H.T @ x  # (m, in_dim)

        # Stage 2: Hyperedge -> Vertex (sum aggregation)
        agg = H @ e  # (n, in_dim)

        # Combine with self-loop weighted by (1 + epsilon)
        out = (1.0 + self.epsilon) * x + agg

        # Apply MLP
        out = jax.vmap(self.mlp)(out)

        # Zero out isolated nodes (degree 0)
        d_v = jnp.sum(H, axis=1, keepdims=True)
        out = jnp.where(d_v > 0, out, 0.0)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
