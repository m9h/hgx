"""Lorentz (hyperboloid) model hypergraph convolution.

Implements hypergraph message passing in the Lorentz model of hyperbolic
space, where points live on the upper sheet of the hyperboloid:

    H^d_c = {x in R^{d+1} : <x, x>_L = -1/c,  x_0 > 0}

The Minkowski inner product is <x, y>_L = -x_0*y_0 + x_1*y_1 + ... + x_d*y_d.

Key operations:
  - **Lorentz linear**: tangent-space linear map followed by exp_0
  - **Einstein midpoint**: centroid on the hyperboloid for aggregation
  - **Learnable curvature**: scalar parameter c > 0

References:
    Chami et al. (2019). "Hyperbolic Graph Convolutional Neural Networks."
    NeurIPS 2019.

    Law & Stam (2020). "Lorentz Equivariant Model."

    Chen et al. (2022). "Fully Hyperbolic Neural Networks." ACL 2022.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# Lorentz geometry primitives
# ---------------------------------------------------------------------------

def lorentz_inner(
    x: Float[Array, "... d"], y: Float[Array, "... d"],
) -> Float[Array, ...]:
    """Minkowski inner product <x, y>_L = -x_0*y_0 + sum_{i>0} x_i*y_i."""
    return -x[..., 0] * y[..., 0] + jnp.sum(x[..., 1:] * y[..., 1:], axis=-1)


def lorentz_norm_sq(x: Float[Array, "... d"]) -> Float[Array, ...]:
    """Squared Lorentz norm <x, x>_L."""
    return lorentz_inner(x, x)


def project_to_hyperboloid(
    x: Float[Array, "... d"], c: Float[Array, ""],
) -> Float[Array, "... d"]:
    """Project a point onto the hyperboloid H^d_c.

    Given spatial components x[..., 1:], compute x_0 so that
    <x, x>_L = -1/c.  That is, x_0 = sqrt(1/c + ||x_space||^2).
    """
    space = x[..., 1:]
    space_sq = jnp.sum(space ** 2, axis=-1, keepdims=True)
    x0 = jnp.sqrt(jnp.clip(1.0 / c + space_sq, min=1e-8))
    return jnp.concatenate([x0, space], axis=-1)


def exp_map_0(v: Float[Array, "... d"], c: Float[Array, ""]) -> Float[Array, "... d"]:
    """Exponential map at the origin of H^d_c.

    The origin on the hyperboloid is o = (1/sqrt(c), 0, ..., 0).
    For a tangent vector v at o (where v_0 = 0):

        exp_o(v) = cosh(sqrt(c)*||v||) * o + sinh(sqrt(c)*||v||)/||v|| * v

    Args:
        v: Tangent vectors at the origin. v[..., 0] is ignored / set to 0.
        c: Positive curvature parameter.

    Returns:
        Points on the hyperboloid.
    """
    sqrt_c = jnp.sqrt(c)
    # Only the spatial part matters for tangent vectors at origin
    v_space = v[..., 1:]
    v_sq = jnp.sum(v_space ** 2, axis=-1, keepdims=True)
    v_norm = jnp.sqrt(jnp.clip(v_sq, min=1e-10))

    coeff_o = jnp.cosh(sqrt_c * v_norm)  # coefficient for origin
    coeff_v = jnp.sinh(sqrt_c * v_norm) / (sqrt_c * v_norm + 1e-10)  # coefficient for v

    # Origin: (1/sqrt(c), 0, ..., 0)
    x0 = coeff_o / sqrt_c  # cosh(...) * 1/sqrt(c)
    x_space = coeff_v * v_space

    return jnp.concatenate([x0, x_space], axis=-1)


def log_map_0(x: Float[Array, "... d"], c: Float[Array, ""]) -> Float[Array, "... d"]:
    """Logarithmic map at the origin of H^d_c.

    Inverse of exp_map_0.  Returns tangent vectors at the origin.

    Args:
        x: Points on the hyperboloid.
        c: Positive curvature parameter.

    Returns:
        Tangent vectors at the origin (with v_0 = 0).
    """
    sqrt_c = jnp.sqrt(c)
    x0 = x[..., :1]
    x_space = x[..., 1:]

    # d_L(o, x) = acosh(-sqrt(c) * x_0)  (scaled)
    arg = jnp.clip(sqrt_c * x0, min=1.0 + 1e-7)
    dist = jnp.arccosh(arg)

    x_sq = jnp.sum(x_space ** 2, axis=-1, keepdims=True)
    x_space_norm = jnp.sqrt(jnp.clip(x_sq, min=1e-10))
    coeff = dist / (sqrt_c * x_space_norm + 1e-10)

    v_space = coeff * x_space
    v0 = jnp.zeros_like(x0)
    return jnp.concatenate([v0, v_space], axis=-1)


def einstein_midpoint(
    x: Float[Array, "k d"],
    weights: Float[Array, " k"] | None = None,
    c: Float[Array, ""] = jnp.array(1.0),
) -> Float[Array, " d"]:
    """Einstein midpoint (weighted centroid) on the hyperboloid.

    For points x_1, ..., x_k on H^d_c with weights w_i:

        gamma_i = 1 / sqrt(|<x_i, x_i>_L|)
        (Lorentz factor, = sqrt(c) for on-manifold points)

    The Klein midpoint in the Klein model is:
        m_K = sum(w_i * gamma_i * x_i) / sum(w_i * gamma_i)

    Then project back to the hyperboloid.

    For points exactly on the hyperboloid with curvature c,
    gamma_i = sqrt(c) * x_i_0 for the time component.

    Args:
        x: Points on the hyperboloid, shape (k, d).
        weights: Optional per-point weights, shape (k,).
        c: Curvature parameter.

    Returns:
        Centroid on the hyperboloid, shape (d,).
    """
    # Lorentz factor: gamma_i = sqrt(c) * x_0
    # But more numerically robust: gamma_i = 1/sqrt(|<x_i,x_i>_L + 1/c|) ...
    # For on-manifold points <x,x>_L = -1/c, so gamma simplifies.
    # We use the standard formula: gamma_i proportional to x_0 (time component).
    gamma = jnp.sqrt(jnp.clip(c, min=1e-8)) * x[..., 0]  # (k,)

    if weights is not None:
        gamma = gamma * weights

    # Weighted sum in ambient space
    denom = jnp.sum(gamma) + 1e-10
    midpoint = jnp.sum(gamma[:, None] * x, axis=0) / denom

    # Project back onto hyperboloid
    return project_to_hyperboloid(midpoint, c)


# ---------------------------------------------------------------------------
# Lorentz Hypergraph Convolution
# ---------------------------------------------------------------------------

class LorentzHypergraphConv(AbstractHypergraphConv):
    """Hypergraph convolution in the Lorentz (hyperboloid) model.

    Message passing operates natively in hyperbolic space:

    1. **Log map**: map node features from the hyperboloid to the tangent
       space at the origin.
    2. **Linear transform**: apply a learnable linear map in tangent space.
    3. **Exp map**: map back to the hyperboloid.
    4. **Vertex -> Hyperedge**: aggregate member vertices via the Einstein
       midpoint to produce hyperedge representations.
    5. **Hyperedge -> Vertex**: aggregate incident hyperedge centroids back
       to each vertex, again via Einstein midpoint.

    The layer activates when ``hg.geometry == "lorentz"``.  If the geometry
    is not ``"lorentz"``, features are treated as ambient-space coordinates
    and projected onto the hyperboloid before processing.

    Node features must have dimension ``d + 1`` where ``d`` is the
    hyperbolic dimension (the first component is the time coordinate).

    Attributes:
        linear: Tangent-space linear map (operates on spatial dimensions).
        log_c: Log of the curvature parameter (unconstrained; c = exp(log_c)).
        use_bias: Whether the linear layer includes a bias.
    """

    linear: eqx.nn.Linear
    log_c: Float[Array, ""]
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        init_curvature: float = 1.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize LorentzHypergraphConv.

        ``in_dim`` and ``out_dim`` refer to the **full ambient dimension**
        (d+1), where the first coordinate is the time component.  The
        internal linear map operates on the d spatial dimensions, so it
        maps (in_dim - 1) -> (out_dim - 1).

        Args:
            in_dim: Input ambient dimension (d_in + 1).
            out_dim: Output ambient dimension (d_out + 1).
            use_bias: Whether to include bias in the tangent-space linear.
            init_curvature: Initial value of the curvature c > 0.
            key: PRNG key for weight initialization.
        """
        if in_dim < 2 or out_dim < 2:
            raise ValueError(
                f"in_dim and out_dim must be >= 2 (ambient dim = spatial + 1). "
                f"Got in_dim={in_dim}, out_dim={out_dim}."
            )
        # Linear operates on spatial components only (d = dim - 1)
        self.linear = eqx.nn.Linear(in_dim - 1, out_dim - 1, use_bias=use_bias, key=key)
        self.log_c = jnp.log(jnp.array(init_curvature, dtype=jnp.float32))
        self.use_bias = use_bias

    @property
    def c(self) -> Float[Array, ""]:
        """Positive curvature parameter."""
        return jnp.exp(self.log_c)

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply Lorentz hypergraph convolution.

        Args:
            hg: Input hypergraph. Node features have shape (n, in_dim).
                If ``hg.geometry == "lorentz"``, features are assumed to
                already lie on the hyperboloid.  Otherwise they are
                projected first.

        Returns:
            Updated node features of shape (n, out_dim), lying on the
            hyperboloid H^{out_dim-1}_c.
        """
        H = hg._masked_incidence()  # (n, m)
        c = self.c
        x = hg.node_features  # (n, in_dim)

        # Ensure points lie on the hyperboloid
        if hg.geometry != "lorentz":
            x = project_to_hyperboloid(x, c)

        # --- Tangent-space linear transform ---
        # Log map to tangent space at origin
        v = log_map_0(x, c)  # (n, in_dim), v[..., 0] ≈ 0

        # Apply linear to spatial components
        v_space = v[..., 1:]  # (n, in_dim - 1)
        v_out = jax.vmap(self.linear)(v_space)  # (n, out_dim - 1)

        # Prepend zero time component and exp map back
        v_full = jnp.concatenate([jnp.zeros((x.shape[0], 1)), v_out], axis=-1)
        x_transformed = exp_map_0(v_full, c)  # (n, out_dim)

        # --- Stage 1: Vertex -> Hyperedge via Einstein midpoint ---
        n, m = H.shape
        # Compute hyperedge degree for weighting

        # For each hyperedge k, compute Einstein midpoint of member vertices.
        # We vectorize over hyperedges: use membership weights from H.
        def _edge_centroid(col: Float[Array, " n"]) -> Float[Array, " out_d"]:
            """Einstein midpoint of vertices in one hyperedge."""
            # col is H[:, k] — membership weights for hyperedge k
            return einstein_midpoint(x_transformed, weights=col, c=c)

        e = jax.vmap(_edge_centroid, in_axes=1)(H)  # (m, out_d)

        # --- Stage 2: Hyperedge -> Vertex via Einstein midpoint ---
        def _vertex_agg(row: Float[Array, " m"]) -> Float[Array, " out_d"]:
            """Einstein midpoint of incident hyperedges for one vertex."""
            return einstein_midpoint(e, weights=row, c=c)

        out = jax.vmap(_vertex_agg, in_axes=0)(H)  # (n, out_d)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
