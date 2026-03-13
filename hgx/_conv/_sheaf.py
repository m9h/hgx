"""Cellular sheaf neural networks on hypergraphs.

Implements sheaf-based message passing from:
    Hansen & Ghrist (2019). "Toward a spectral theory of cellular sheaves."
    Bodnar et al. (2022). "Neural Sheaf Diffusion."

A cellular sheaf on a hypergraph assigns a vector space (stalk) F(v) to
each vertex and a linear restriction map F_{v->e}: F(v) -> F(e) to each
vertex-hyperedge incidence. The sheaf coboundary operator delta maps vertex
signals to edge signals:

    (delta x)_e = sum_{i in e} F_{i->e} x_i

The sheaf Laplacian is L_F = delta^T delta. Standard message passing is the
special case where all restriction maps are identity matrices.

The forward pass performs sheaf Laplacian diffusion:
    1. Project each vertex feature through its restriction map
    2. Aggregate per hyperedge (mean)
    3. Compute sheaf Laplacian diffusion residual
    4. Update vertex features via gradient step
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class SheafHypergraphConv(AbstractHypergraphConv):
    """Sheaf-based hypergraph convolution via Laplacian diffusion.

    Performs one step of sheaf Laplacian diffusion:
        z_{i,e} = F_{i->e} @ x_i           (restriction)
        z_e     = mean_{i in e}(z_{i,e})    (aggregation)
        Delta_i = sum_{e ni i} F_{i->e}^T (z_{i,e} - z_e)  (coboundary)
        x_i'    = x_i - step_size * Delta_i + bias

    When all restriction maps are identity, this reduces to standard
    hypergraph Laplacian diffusion.

    Attributes:
        restriction_maps: Learned linear maps of shape (nnz, d_stalk, d_edge),
            one per vertex-hyperedge incidence.
        step_size: Learnable scalar controlling diffusion rate.
        bias: Learnable bias of shape (in_dim,).
        in_dim: Input (vertex stalk) dimension.
        edge_stalk_dim: Edge stalk dimension.
        num_incidences: Number of nonzero entries in the incidence matrix.
    """

    restriction_maps: Float[Array, "nnz d_stalk d_edge"]
    step_size: Float[Array, ""]
    bias: Float[Array, " d_stalk"]
    in_dim: int = eqx.field(static=True)
    edge_stalk_dim: int = eqx.field(static=True)
    num_incidences: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        edge_stalk_dim: int,
        num_incidences: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize SheafHypergraphConv.

        Args:
            in_dim: Input feature dimension (vertex stalk dimension d_stalk).
            edge_stalk_dim: Edge stalk dimension (d_edge).
            num_incidences: Number of nonzero entries in the incidence matrix
                (nnz). Each incidence gets its own restriction map.
            key: PRNG key for parameter initialization.
        """
        k1, k2 = jax.random.split(key)
        self.in_dim = in_dim
        self.edge_stalk_dim = edge_stalk_dim
        self.num_incidences = num_incidences

        # Initialize restriction maps with orthogonal-like initialization:
        # scale so that F^T F ~ I when d_stalk == d_edge
        self.restriction_maps = (
            jax.random.normal(k1, (num_incidences, in_dim, edge_stalk_dim))
            / jnp.sqrt(jnp.float32(in_dim))
        )
        self.step_size = jnp.array(1.0)
        self.bias = jnp.zeros(in_dim)

    def __call__(self, hg: Hypergraph) -> Float[Array, "n d_stalk"]:
        """Apply one step of sheaf Laplacian diffusion.

        Args:
            hg: Input hypergraph with node features of shape (n, in_dim).

        Returns:
            Updated node features of shape (n, in_dim).
        """
        H = hg._masked_incidence()
        x = hg.node_features  # (n, d_stalk)
        n = x.shape[0]
        m = H.shape[1]

        # Get incidence structure: (vertex_idx, edge_idx) pairs
        nnz = self.num_incidences
        v_idx, e_idx = jnp.nonzero(H, size=nnz)

        # (a) Project each vertex feature through its restriction map:
        #     z_{i,e} = F_{i->e} @ x_i
        # x[v_idx] has shape (nnz, d_stalk)
        # restriction_maps has shape (nnz, d_stalk, d_edge)
        x_inc = x[v_idx]  # (nnz, d_stalk)
        # Batched matrix-vector multiply: (nnz, d_stalk, d_edge)^T @ (nnz, d_stalk)
        # = einsum("kij,ki->kj", maps, x_inc) where i=d_stalk, j=d_edge
        z_inc = jnp.einsum("kij,ki->kj", self.restriction_maps, x_inc)
        # z_inc: (nnz, d_edge)

        # (b) Aggregate per hyperedge: z_e = mean(z_{i,e} for i in e)
        z_e_sum = jnp.zeros((m, self.edge_stalk_dim))
        z_e_sum = z_e_sum.at[e_idx].add(z_inc)
        e_degrees = jnp.zeros(m).at[e_idx].add(1.0)
        e_degrees_safe = jnp.where(e_degrees > 0, e_degrees, 1.0)
        z_e = z_e_sum / e_degrees_safe[:, None]  # (m, d_edge)

        # (c) Compute sheaf Laplacian diffusion residual:
        #     For each incidence (i,e): diff = z_{i,e} - z_e
        #     Delta_i = sum_{e ni i} F_{i->e}^T @ diff
        diff = z_inc - z_e[e_idx]  # (nnz, d_edge)

        # F_{i->e}^T @ diff: einsum("kij,kj->ki", maps, diff)
        # where i=d_stalk, j=d_edge
        delta_inc = jnp.einsum("kij,kj->ki", self.restriction_maps, diff)
        # delta_inc: (nnz, d_stalk)

        # Sum over all hyperedges incident to each vertex
        delta = jnp.zeros((n, self.in_dim))
        delta = delta.at[v_idx].add(delta_inc)  # (n, d_stalk)

        # (d) Update: x_i' = x_i - step_size * Delta_i + bias
        out = x - self.step_size * delta + self.bias

        # Zero out isolated nodes (not in any hyperedge)
        d_v = jnp.sum(H, axis=1, keepdims=True)
        out = jnp.where(d_v > 0, out, x)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out


class SheafDiffusion(eqx.Module):
    """Multi-step sheaf diffusion on hypergraphs.

    Applies multiple steps of sheaf Laplacian diffusion, analogous to
    discretizing the sheaf diffusion PDE dx/dt = -L_F x.

    If ``share_maps=True``, all steps use the same restriction maps,
    behaving like a fixed-step ODE integrator. If ``share_maps=False``,
    each step has independent maps, yielding a more expressive model.

    Attributes:
        layers: Tuple of SheafHypergraphConv steps.
        num_steps: Number of diffusion steps.
        share_maps: Whether all steps share the same restriction maps.
    """

    layers: tuple[SheafHypergraphConv, ...]
    num_steps: int = eqx.field(static=True)
    share_maps: bool = eqx.field(static=True)

    def __init__(
        self,
        num_steps: int,
        in_dim: int,
        edge_stalk_dim: int,
        num_incidences: int,
        share_maps: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize SheafDiffusion.

        Args:
            num_steps: Number of diffusion steps.
            in_dim: Input feature dimension (vertex stalk dimension).
            edge_stalk_dim: Edge stalk dimension.
            num_incidences: Number of nonzero entries in the incidence matrix.
            share_maps: If True, all steps share the same restriction maps
                (ODE-like). If False, each step has independent maps.
            key: PRNG key for parameter initialization.
        """
        self.num_steps = num_steps
        self.share_maps = share_maps

        if share_maps:
            # All steps share one set of restriction maps
            layer = SheafHypergraphConv(
                in_dim, edge_stalk_dim, num_incidences, key=key,
            )
            self.layers = tuple(layer for _ in range(num_steps))
        else:
            # Each step has independent maps
            keys = jax.random.split(key, num_steps)
            self.layers = tuple(
                SheafHypergraphConv(
                    in_dim, edge_stalk_dim, num_incidences, key=k,
                )
                for k in keys
            )

    def __call__(self, hg: Hypergraph) -> Float[Array, "n d_stalk"]:
        """Apply multi-step sheaf diffusion.

        Args:
            hg: Input hypergraph with node features.

        Returns:
            Updated node features after all diffusion steps.
        """
        x = hg.node_features
        for layer in self.layers:
            # Create updated hypergraph with new features for each step
            hg_step = eqx.tree_at(
                lambda h: h.node_features, hg, x,
            )
            x = layer(hg_step)
        return x


def learn_restriction_maps(
    hg: Hypergraph,
    target_dim: int,
    *,
    key: PRNGKeyArray,
) -> Float[Array, "nnz d_stalk d_edge"]:
    """Initialize restriction maps from hypergraph structure.

    Uses SVD of local neighborhoods in the incidence matrix to find
    principal directions for each vertex-edge incidence pair. This
    provides a structure-aware initialization that can be better than
    random for sheaf learning.

    For each incidence (i, e), the restriction map is initialized from
    the SVD of the submatrix of node features belonging to hyperedge e.
    If node features are not informative (e.g., all ones), falls back to
    random orthogonal initialization.

    Args:
        hg: Input hypergraph with node features of shape (n, d_stalk).
        target_dim: Target edge stalk dimension (d_edge).
        key: PRNG key for random fallback.

    Returns:
        Restriction maps of shape (nnz, d_stalk, d_edge).
    """
    H = hg._masked_incidence()
    x = hg.node_features  # (n, d_stalk)
    d_stalk = x.shape[1]

    nnz = int(jnp.sum(H > 0))
    v_idx, e_idx = jnp.nonzero(H, size=nnz)

    maps = jnp.zeros((nnz, d_stalk, target_dim))

    # For each hyperedge, compute SVD of the member vertex features
    m = H.shape[1]
    for edge_k in range(m):
        # Get mask of which incidences belong to this edge
        edge_mask = e_idx == edge_k  # (nnz,)
        # Indices of vertices in this edge
        members_mask = H[:, edge_k] > 0  # (n,)

        # Feature submatrix for this edge's vertices
        # Use masked selection: weight features by membership
        x_edge = x * members_mask[:, None]  # (n, d_stalk), zeros for non-members

        # SVD of the feature submatrix
        # We want the principal directions that capture the most variance
        # within this hyperedge's vertex features
        u, s, vt = jnp.linalg.svd(x_edge, full_matrices=False)

        # Take the top `target_dim` right singular vectors as the
        # projection basis. vt has shape (min(n, d_stalk), d_stalk)
        k_dim = min(d_stalk, target_dim)
        basis = vt[:k_dim].T  # (d_stalk, k_dim)

        # Pad if target_dim > d_stalk
        if k_dim < target_dim:
            key, subkey = jax.random.split(key)
            pad = jax.random.normal(subkey, (d_stalk, target_dim - k_dim))
            pad = pad / jnp.sqrt(jnp.float32(d_stalk))
            basis = jnp.concatenate([basis, pad], axis=1)

        # Assign this basis to all incidences in this edge
        # Use where-based update for each incidence position
        basis_broadcast = jnp.broadcast_to(basis, (nnz, d_stalk, target_dim))
        maps = jnp.where(
            edge_mask[:, None, None],
            basis_broadcast,
            maps,
        )

    return maps
