"""Tensorized Hypergraph Neural Network (THNN) convolution layer.

Implements the THNN layer from:
    Wang et al. (2024). "Tensorized Hypergraph Neural Networks." SDM 2024.
    arXiv:2306.02560

Unlike first-order methods (UniGCN) that aggregate via summation — losing
the joint interaction structure — THNN uses element-wise products of
projected features to capture genuine multilinear (high-order) interactions
within each hyperedge.

The key insight: for a hyperedge {j1, j2, j3}, the interaction
    z_j1 * z_j2 * z_j3  (element-wise product)
captures the joint effect of all three vertices simultaneously,
whereas sum aggregation only captures their independent contributions.

Partially symmetric CP decomposition keeps parameters linear in rank R
rather than exponential in hyperedge order.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class THNNConv(AbstractHypergraphConv):
    """Tensorized Hypergraph Neural Network convolution.

    For each hyperedge e = {j1, ..., jk}:
        1. Project: z_ji = Theta^T [h_ji ; 1]  (shared projection to rank R)
        2. Interact: m_e = z_j1 * z_j2 * ... * z_jk  (element-wise product)
        3. Output: contribution_i += Q * tanh(m_e)

    The concatenation of [h; 1] ensures lower-order polynomial terms
    are preserved (section 3.4 of the paper).

    Attributes:
        theta: Shared projection matrix (in_dim+1, rank).
        q: Output projection matrix (rank, out_dim).
        rank: CP decomposition rank.
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
        """Initialize THNNConv.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            rank: CP decomposition rank R. Controls expressiveness vs cost.
            normalize: Whether to apply degree normalization.
            key: PRNG key for weight initialization.
        """
        k1, k2 = jax.random.split(key)
        # +1 for the appended constant (the [h; 1] trick)
        self.theta = eqx.nn.Linear(in_dim + 1, rank, use_bias=False, key=k1)
        self.q = eqx.nn.Linear(rank, out_dim, use_bias=True, key=k2)
        self.rank = rank
        self.normalize = normalize

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply THNN convolution.

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

        # For each hyperedge, compute element-wise product of member projections.
        # We use log-sum-exp trick: prod(z_j) = exp(sum(log(|z_j|))) * sign
        # But for simplicity and numerical stability with tanh downstream,
        # we compute via log-space aggregation over the incidence structure.
        #
        # Direct approach: for each hyperedge k, multiply the z vectors
        # of its member vertices. Using log-domain:
        #   log(|m_k|) = H^T @ log(|z|)   (sum of logs = log of product)
        #   sign(m_k) = product of signs = exp(H^T @ log(sign))
        #
        # This is equivalent to the tensor contraction but O(nnz * rank).

        # Numerical stable product aggregation via log domain
        eps = 1e-8
        z_abs = jnp.abs(z) + eps
        z_sign = jnp.sign(z)

        # Sum of logs for magnitude
        log_prod = H.T @ jnp.log(z_abs)  # (m, rank)

        # Product of signs: use the fact that sign is +/-1
        # log(sign) trick: encode sign as 0 or pi, sum, take cos
        sign_angle = jnp.where(z_sign < 0, jnp.pi, 0.0)
        total_angle = H.T @ sign_angle  # (m, rank)
        prod_sign = jnp.cos(total_angle)  # +1 or -1

        # Reconstruct product
        m_e = prod_sign * jnp.exp(log_prod)  # (m, rank)

        # Apply tanh nonlinearity and output projection
        m_e = jnp.tanh(m_e)
        e_out = jax.vmap(self.q)(m_e)  # (m, out_dim)

        # Degree normalization
        if self.normalize:
            d_v = jnp.sum(H, axis=1, keepdims=True)  # (n, 1)
            d_v_inv = jnp.where(d_v > 0, 1.0 / d_v, 0.0)
            out = d_v_inv * (H @ e_out)
        else:
            out = H @ e_out  # (n, out_dim)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
