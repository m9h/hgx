"""Sparse HGNN convolution using segment_sum-based message passing.

Drop-in replacement for HGNNConv that uses segment_sum over
star-expansion indices instead of dense matrix multiplication,
reducing complexity from O(n*m) to O(nnz). Also accepts
SparseHypergraph natively.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph
from hgx._sparse import edge_to_vertex, incidence_to_star_expansion, vertex_to_edge
from hgx._sparse_incidence import (
    SparseHypergraph,
    sparse_edge_to_vertex,
    sparse_vertex_to_edge,
    to_sparse,
)


class HGNNSparseConv(AbstractHypergraphConv):
    """Sparse HGNN: propagate first, project after (O(nnz) complexity).

    Accepts either a dense Hypergraph (converted internally via
    star-expansion) or a SparseHypergraph (uses COO segment_sum).

    Attributes:
        linear: Linear projection applied after aggregation.
        normalize: Whether to apply symmetric degree normalization.
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

    def __call__(
        self, hg: Hypergraph | SparseHypergraph
    ) -> Float[Array, "n out_dim"]:
        """Apply sparse HGNN convolution.

        Args:
            hg: Input hypergraph (dense or sparse).

        Returns:
            Updated node features of shape (num_nodes, out_dim).
        """
        if isinstance(hg, SparseHypergraph):
            return self._forward_sparse(hg)
        return self._forward_dense_star(hg)

    def _forward_dense_star(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Forward using star-expansion of dense incidence."""
        H = hg._masked_incidence()
        n, m = H.shape
        x = hg.node_features  # (n, in_dim)

        v_idx, e_idx, valid = incidence_to_star_expansion(H)

        if self.normalize:
            d_e = jnp.sum(H, axis=0)  # (m,)
            d_v = jnp.sum(H, axis=1)  # (n,)

            d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
            d_v_inv_sqrt = jnp.where(d_v > 0, 1.0 / jnp.sqrt(d_v), 0.0)

            # D_v^{-1/2} x
            x_scaled = d_v_inv_sqrt[:, None] * x

            # V -> E with edge-degree normalization
            e = vertex_to_edge(x_scaled, v_idx, e_idx, m, valid) * d_e_inv[:, None]

            # E -> V
            x_smooth = edge_to_vertex(e, v_idx, e_idx, n, valid)

            # D_v^{-1/2}
            x_smooth = d_v_inv_sqrt[:, None] * x_smooth
        else:
            e = vertex_to_edge(x, v_idx, e_idx, m, valid)
            x_smooth = edge_to_vertex(e, v_idx, e_idx, n, valid)

        out = jax.vmap(self.linear)(x_smooth)

        # Zero out isolated nodes
        d_v_mask = jnp.sum(H, axis=1)
        active = (d_v_mask > 0).astype(out.dtype)
        out = out * active[:, None]

        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out

    def _forward_sparse(self, hg: SparseHypergraph) -> Float[Array, "n out_dim"]:
        """Forward using native sparse incidence (COO segment_sum)."""
        indices, masked_data = hg._masked_incidence_sparse()
        n, m = hg.shape
        x = hg.node_features  # (n, in_dim)

        # Compute vertex degrees for isolated node masking
        d_v = jax.ops.segment_sum(masked_data, indices[:, 0], num_segments=n)

        if self.normalize:
            d_e = jax.ops.segment_sum(masked_data, indices[:, 1], num_segments=m)

            d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
            d_v_inv_sqrt = jnp.where(d_v > 0, 1.0 / jnp.sqrt(d_v), 0.0)

            # D_v^{-1/2} x
            x_scaled = d_v_inv_sqrt[:, None] * x

            # V -> E with edge-degree normalization
            e = sparse_vertex_to_edge(x_scaled, indices, masked_data, (n, m))
            e = e * d_e_inv[:, None]

            # E -> V
            x_smooth = sparse_edge_to_vertex(e, indices, masked_data, (n, m))

            # D_v^{-1/2}
            x_smooth = d_v_inv_sqrt[:, None] * x_smooth
        else:
            e = sparse_vertex_to_edge(x, indices, masked_data, (n, m))
            x_smooth = sparse_edge_to_vertex(e, indices, masked_data, (n, m))

        out = jax.vmap(self.linear)(x_smooth)

        # Zero out isolated nodes
        active = (d_v > 0).astype(out.dtype)
        out = out * active[:, None]

        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
