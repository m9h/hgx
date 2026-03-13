"""UniGCN convolution layer for hypergraphs.

Implements the UniGCN variant from:
    Huang & Yang (2021). "UniGNN: a Unified Framework for Graph and
    Hypergraph Neural Networks." IJCAI 2021.

The layer performs two-stage message passing:
    1. Vertex -> Hyperedge: aggregate vertex features to hyperedges
    2. Hyperedge -> Vertex: aggregate hyperedge messages back to vertices

When applied to a 2-uniform hypergraph (all hyperedges have exactly
2 vertices), this reduces to standard GCN.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class UniGCNConv(AbstractHypergraphConv):
    """UniGCN: first-order hypergraph convolution via sum aggregation.

    Performs symmetric-normalized two-stage message passing:
        e_k = (1/|e_k|) * sum_{i in e_k} (W * h_i)
        h_i = (1/d_i) * sum_{k: i in e_k} e_k

    where |e_k| is the hyperedge degree and d_i is the vertex degree.

    Attributes:
        linear: Linear projection applied to node features before aggregation.
        use_bias: Whether the linear layer includes a bias term.
        normalize: Whether to apply degree normalization.
    """

    linear: eqx.nn.Linear
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        normalize: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize UniGCNConv.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            use_bias: Whether to include bias in the linear layer.
            normalize: Whether to apply symmetric degree normalization.
            key: PRNG key for weight initialization.
        """
        self.linear = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=key)
        self.normalize = normalize

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply UniGCN convolution.

        Args:
            hg: Input hypergraph.

        Returns:
            Updated node features of shape (num_nodes, out_dim).
        """
        H = hg._masked_incidence()
        x = jax.vmap(self.linear)(hg.node_features)

        if self.normalize:
            # Degree normalization
            d_v = jnp.sum(H, axis=1, keepdims=True)  # (n, 1) vertex degrees
            d_e = jnp.sum(H, axis=0, keepdims=True)  # (1, m) hyperedge degrees

            # Avoid division by zero
            d_v_inv = jnp.where(d_v > 0, 1.0 / d_v, 0.0)
            d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)

            # Stage 1: Vertex -> Hyperedge (with edge degree normalization)
            e = (H * d_e_inv).T @ x  # (m, out_dim)

            # Stage 2: Hyperedge -> Vertex (with vertex degree normalization)
            out = (H * d_v_inv) @ e  # (n, out_dim) -- note: d_v_inv broadcasts
            # Correct: apply d_v_inv after aggregation
            out = d_v_inv * (H @ e)
        else:
            # Unnormalized
            e = H.T @ x       # (m, out_dim) vertex -> hyperedge
            out = H @ e        # (n, out_dim) hyperedge -> vertex

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
