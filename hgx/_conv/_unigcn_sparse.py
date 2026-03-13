"""Sparse UniGCN convolution using index-based message passing.

Drop-in replacement for UniGCNConv that uses segment_sum over
star-expansion indices instead of dense matrix multiplication,
reducing complexity from O(n*m) to O(nnz).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph
from hgx._sparse import edge_to_vertex, incidence_to_star_expansion, vertex_to_edge


class UniGCNSparseConv(AbstractHypergraphConv):
    """Sparse UniGCN: segment_sum-based hypergraph convolution.

    Numerically equivalent to UniGCNConv but uses index-based
    aggregation instead of dense matrix multiplication.

    Attributes:
        linear: Linear projection applied to node features before aggregation.
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
        self.linear = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=key)
        self.normalize = normalize

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        H = hg._masked_incidence()
        n, m = H.shape

        x = jax.vmap(self.linear)(hg.node_features)

        v_idx, e_idx, valid = incidence_to_star_expansion(H)

        if self.normalize:
            d_e = jnp.sum(H, axis=0)  # (m,)
            d_v = jnp.sum(H, axis=1)  # (n,)

            d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
            d_v_inv = jnp.where(d_v > 0, 1.0 / d_v, 0.0)

            # Stage 1: V→E with edge-degree normalization
            e = vertex_to_edge(x, v_idx, e_idx, m, valid) * d_e_inv[:, None]

            # Stage 2: E→V with vertex-degree normalization
            out = edge_to_vertex(e, v_idx, e_idx, n, valid) * d_v_inv[:, None]
        else:
            e = vertex_to_edge(x, v_idx, e_idx, m, valid)
            out = edge_to_vertex(e, v_idx, e_idx, n, valid)

        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
