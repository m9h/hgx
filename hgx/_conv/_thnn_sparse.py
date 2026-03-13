"""Sparse Tensorized Hypergraph Neural Network (THNN) convolution layer.

Drop-in replacement for THNNConv that uses segment_sum over
star-expansion indices instead of dense matrix multiplication,
reducing complexity from O(n*m) to O(nnz).

The product interaction is computed in log-domain via segment_sum:
    log_prod_per_edge = segment_sum(log(|z|)[v_idx], e_idx)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph
from hgx._sparse import edge_to_vertex, incidence_to_star_expansion


class THNNSparseConv(AbstractHypergraphConv):
    """Sparse THNN: segment_sum-based tensorized hypergraph convolution.

    Numerically equivalent to THNNConv but uses index-based
    aggregation instead of dense matrix multiplication.

    Attributes:
        theta: Shared projection matrix (in_dim+1, rank).
        q: Output projection matrix (rank, out_dim).
        rank: CP decomposition rank.
        normalize: Whether to apply degree normalization.
    """

    theta: eqx.nn.Linear
    q: eqx.nn.Linear
    rank: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 64,
        normalize: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize THNNSparseConv.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            rank: CP decomposition rank R.
            normalize: Whether to apply degree normalization.
            key: PRNG key for weight initialization.
        """
        k1, k2 = jax.random.split(key)
        self.theta = eqx.nn.Linear(in_dim + 1, rank, use_bias=False, key=k1)
        self.q = eqx.nn.Linear(rank, out_dim, use_bias=True, key=k2)
        self.rank = rank
        self.normalize = normalize

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply sparse THNN convolution.

        Args:
            hg: Input hypergraph.

        Returns:
            Updated node features of shape (num_nodes, out_dim).
        """
        H = hg._masked_incidence()
        n, m = H.shape

        # Append constant 1 to features: [h_i; 1]
        ones = jnp.ones((n, 1))
        x_aug = jnp.concatenate([hg.node_features, ones], axis=1)

        # Shared projection: z_i = Theta^T [h_i; 1], shape (n, rank)
        z = jax.vmap(self.theta)(x_aug)

        # Star expansion indices
        v_idx, e_idx, valid = incidence_to_star_expansion(H)

        # Product aggregation via log domain using segment_sum
        eps = 1e-8
        z_abs = jnp.abs(z) + eps
        z_sign = jnp.sign(z)

        # Sum of logs for magnitude (sparse, masked by valid)
        log_z = jnp.log(z_abs)  # (n, rank)
        log_vals = log_z[v_idx] * valid[:, None]
        log_prod = jax.ops.segment_sum(
            log_vals, e_idx, num_segments=m
        )  # (m, rank)

        # Product of signs via angle trick (sparse, masked by valid)
        sign_angle = jnp.where(z_sign < 0, jnp.pi, 0.0)  # (n, rank)
        angle_vals = sign_angle[v_idx] * valid[:, None]
        total_angle = jax.ops.segment_sum(
            angle_vals, e_idx, num_segments=m
        )  # (m, rank)
        prod_sign = jnp.cos(total_angle)

        # Reconstruct product
        m_e = prod_sign * jnp.exp(log_prod)  # (m, rank)

        # Apply tanh nonlinearity and output projection
        m_e = jnp.tanh(m_e)
        e_out = jax.vmap(self.q)(m_e)  # (m, out_dim)

        # Edge -> Vertex aggregation (sparse) with optional normalization
        if self.normalize:
            d_v = jnp.sum(H, axis=1)  # (n,)
            d_v_inv = jnp.where(d_v > 0, 1.0 / d_v, 0.0)
            out = edge_to_vertex(e_out, v_idx, e_idx, n, valid) * d_v_inv[:, None]
        else:
            out = edge_to_vertex(e_out, v_idx, e_idx, n, valid)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
