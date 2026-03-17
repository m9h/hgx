"""HGNN convolution layer: propagate first, project after.

Implements the HGNN convolution from:
    Feng et al. (2019). "Hypergraph Neural Networks." AAAI 2019.

Unlike UniGCNConv which projects features before aggregation
(out = H @ (H^T @ (W @ x))), HGNNConv propagates raw features through
the symmetric normalized Laplacian first, then projects:
    out = W @ (D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} x)

This preserves discriminative signal in high-dimensional inputs
(e.g., Citeseer 3703 features) that early projection would compress.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class HGNNConv(AbstractHypergraphConv):
    """HGNN: propagate first, project after.

    Performs symmetric-normalized Laplacian smoothing followed by
    a linear projection:
        x_smooth = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} x
        out = W @ x_smooth + b

    This matches the original HGNN paper's formulation and differs
    from UniGCNConv which applies the linear projection before aggregation.

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
        """Initialize HGNNConv.

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
        """Apply HGNN convolution.

        Args:
            hg: Input hypergraph.

        Returns:
            Updated node features of shape (num_nodes, out_dim).
        """
        H = hg._masked_incidence()
        x = hg.node_features  # (n, in_dim) — raw features, no projection yet

        if self.normalize:
            d_v = jnp.sum(H, axis=1)  # (n,) vertex degrees
            d_e = jnp.sum(H, axis=0)  # (m,) hyperedge degrees

            d_v_inv_sqrt = jnp.where(d_v > 0, 1.0 / jnp.sqrt(d_v), 0.0)
            d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)

            # Step 1: D_v^{-1/2} x
            x_scaled = d_v_inv_sqrt[:, None] * x  # (n, in_dim)

            # Step 2: H^T @ x_scaled  (vertex -> hyperedge aggregation)
            e = H.T @ x_scaled  # (m, in_dim)

            # Step 3: D_e^{-1} @ e  (edge degree normalization)
            e = d_e_inv[:, None] * e  # (m, in_dim)

            # Step 4: H @ e  (hyperedge -> vertex aggregation)
            x_smooth = H @ e  # (n, in_dim)

            # Step 5: D_v^{-1/2} @ x_smooth
            x_smooth = d_v_inv_sqrt[:, None] * x_smooth  # (n, in_dim)
        else:
            # Unnormalized: just H @ H^T @ x
            e = H.T @ x  # (m, in_dim)
            x_smooth = H @ e  # (n, in_dim)

        # Project AFTER aggregation
        out = jax.vmap(self.linear)(x_smooth)  # (n, out_dim)

        # Zero out isolated nodes (d_v == 0): aggregation gives zero but
        # the linear bias would produce a non-zero output otherwise
        d_v_mask = jnp.sum(H, axis=1)  # (n,)
        active = (d_v_mask > 0).astype(out.dtype)
        out = out * active[:, None]

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
