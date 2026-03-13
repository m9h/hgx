"""UniGAT convolution layer for hypergraphs.

Implements the UniGAT variant from:
    Huang & Yang (2021). "UniGNN: a Unified Framework for Graph and
    Hypergraph Neural Networks." IJCAI 2021.

The layer extends UniGCN with learned attention:
    1. Vertex -> Hyperedge: aggregate vertex features to hyperedges
    2. Compute attention coefficients via LeakyReLU(a^T [Wh_i || e_k])
    3. Hyperedge -> Vertex: attention-weighted aggregation back to vertices
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class UniGATConv(AbstractHypergraphConv):
    """UniGAT: attention-based hypergraph convolution.

    Extends UniGCN by replacing uniform E→V aggregation with learned
    attention weights:
        e_k = (1/|e_k|) * sum_{i in e_k} (W * h_i)
        α_{ik} = softmax_k(LeakyReLU(a^T [Wh_i || e_k]))
        h_i = sum_{k: i in e_k} α_{ik} * e_k

    Attributes:
        linear: Linear projection applied to node features.
        attn: Attention parameter vector of shape (2 * out_dim,).
        negative_slope: LeakyReLU negative slope.
        normalize: Whether to use mean (vs sum) in V→E aggregation.
    """

    linear: eqx.nn.Linear
    attn: Float[Array, " a"]
    negative_slope: float = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        normalize: bool = True,
        negative_slope: float = 0.2,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize UniGATConv.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            use_bias: Whether to include bias in the linear layer.
            normalize: Whether to use mean aggregation in V→E step.
            negative_slope: Negative slope for LeakyReLU on attention.
            key: PRNG key for weight initialization.
        """
        k1, k2 = jax.random.split(key)
        self.linear = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=k1)
        self.attn = jax.random.normal(k2, (2 * out_dim,)) * 0.01
        self.negative_slope = negative_slope
        self.normalize = normalize

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply UniGAT convolution.

        Args:
            hg: Input hypergraph.

        Returns:
            Updated node features of shape (num_nodes, out_dim).
        """
        H = hg._masked_incidence()
        x = jax.vmap(self.linear)(hg.node_features)  # (n, out_dim)

        out_dim = x.shape[-1]
        a_l = self.attn[:out_dim]
        a_r = self.attn[out_dim:]

        # Stage 1: Vertex -> Hyperedge aggregation
        if self.normalize:
            d_e = jnp.sum(H, axis=0, keepdims=True)  # (1, m)
            d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
            e = (H * d_e_inv).T @ x  # (m, out_dim) mean aggregation
        else:
            e = H.T @ x  # (m, out_dim) sum aggregation

        # Compute attention coefficients: LeakyReLU(a^T [Wh_i || e_k])
        v_score = x @ a_l  # (n,)
        e_score = e @ a_r  # (m,)
        raw_attn = v_score[:, None] + e_score[None, :]  # (n, m)

        # LeakyReLU
        raw_attn = jnp.where(
            raw_attn >= 0, raw_attn, self.negative_slope * raw_attn
        )

        # Mask non-incident pairs and apply softmax over hyperedges per vertex
        mask = H > 0
        raw_attn = jnp.where(mask, raw_attn, -1e9)
        attn_weights = jax.nn.softmax(raw_attn, axis=1)  # (n, m)
        attn_weights = attn_weights * H  # zero out non-incident pairs

        # Stage 2: Hyperedge -> Vertex with attention weighting
        out = attn_weights @ e  # (n, out_dim)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
