"""Hyperbolic hypergraph convolution in the Poincaré ball model.

Implements V→E→V message passing on hypergraphs where node features live
in the Poincaré ball (a model of hyperbolic space with constant negative
curvature).  The layer applies a linear transform in the tangent space
at the origin via exponential/logarithmic maps, then aggregates via the
**gyromidpoint** — the natural notion of "average" in hyperbolic geometry
— computed efficiently through the Klein model Einstein midpoint.

Curvature *c* is a learnable parameter (default 1.0).  The ball radius
is 1/√c; larger *c* increases curvature and shrinks the ball.

Intended for use with ``hg.geometry == "poincare"``.

References:
    Chami et al. (2019). "Hyperbolic Graph Convolutional Neural Networks."
        NeurIPS 2019.
    Ganea et al. (2018). "Hyperbolic Neural Networks." NeurIPS 2018.
    Ungar (2008). "Gyrovector Spaces." World Scientific.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


# Numerical stability constants
_EPS = 1e-6
_BALL_EPS = 1e-5


# ---------------------------------------------------------------------------
# Poincaré-ball primitives (pure JAX)
# ---------------------------------------------------------------------------


def project(
    x: Float[Array, "... d"],
    c: Float[Array, ""],
    eps: float = _BALL_EPS,
) -> Float[Array, "... d"]:
    """Project points onto the open Poincaré ball of radius ``1/√c``.

    Clamps the norm to ``1/√c − eps`` so that points remain strictly
    inside the ball boundary.

    Args:
        x: Points in ambient Euclidean space, any leading batch dims.
        c: Positive curvature parameter.
        eps: Safety margin inside the ball boundary.

    Returns:
        Projected points with ``‖x‖ ≤ 1/√c − eps``.
    """
    max_norm = 1.0 / jnp.sqrt(c) - eps
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    factor = jnp.minimum(max_norm / jnp.maximum(norm, _EPS), 1.0)
    return x * factor


def expmap0(
    v: Float[Array, "... d"],
    c: Float[Array, ""],
) -> Float[Array, "... d"]:
    """Exponential map from the origin of the Poincaré ball.

    .. math::

        \\exp_0^c(v) = \\tanh(\\sqrt{c}\\,\\|v\\|)
                       \\;\\frac{v}{\\sqrt{c}\\,\\|v\\|}

    Args:
        v: Tangent vectors at the origin.
        c: Positive curvature parameter.

    Returns:
        Points in the Poincaré ball.
    """
    sqrt_c = jnp.sqrt(c)
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, _EPS)
    return jnp.tanh(sqrt_c * norm) * v / (sqrt_c * norm)


def logmap0(
    x: Float[Array, "... d"],
    c: Float[Array, ""],
) -> Float[Array, "... d"]:
    """Logarithmic map to the origin of the Poincaré ball.

    .. math::

        \\log_0^c(x) = \\operatorname{arctanh}(\\sqrt{c}\\,\\|x\\|)
                       \\;\\frac{x}{\\sqrt{c}\\,\\|x\\|}

    Args:
        x: Points in the Poincaré ball.
        c: Positive curvature parameter.

    Returns:
        Tangent vectors at the origin.
    """
    sqrt_c = jnp.sqrt(c)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, _EPS)
    # Clamp argument of arctanh to stay in (-1, 1)
    sc_norm = jnp.minimum(sqrt_c * norm, 1.0 - _BALL_EPS)
    return jnp.arctanh(sc_norm) * x / (sqrt_c * norm)


def mobius_add(
    x: Float[Array, "... d"],
    y: Float[Array, "... d"],
    c: Float[Array, ""],
) -> Float[Array, "... d"]:
    """Möbius addition ``x ⊕_c y`` in the Poincaré ball.

    .. math::

        x \\oplus_c y = \\frac{(1 + 2c\\langle x,y\\rangle + c\\|y\\|^2)\\,x
        + (1 - c\\|x\\|^2)\\,y}{1 + 2c\\langle x,y\\rangle
        + c^2\\|x\\|^2\\|y\\|^2}

    Args:
        x: First operand in the Poincaré ball.
        y: Second operand in the Poincaré ball.
        c: Positive curvature parameter.

    Returns:
        Result of Möbius addition, in the Poincaré ball.
    """
    xy = jnp.sum(x * y, axis=-1, keepdims=True)
    x_sq = jnp.sum(x ** 2, axis=-1, keepdims=True)
    y_sq = jnp.sum(y ** 2, axis=-1, keepdims=True)

    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c ** 2 * x_sq * y_sq

    return num / jnp.maximum(denom, _EPS)


def gyromidpoint(
    points: Float[Array, "K d"],
    membership: Float[Array, "K G"],
    c: Float[Array, ""],
) -> Float[Array, "G d"]:
    """Gyromidpoint aggregation via the Klein-model Einstein midpoint.

    For each group *g*, computes the Einstein midpoint of the members
    ``{points[k] : membership[k, g] > 0}`` weighted by ``membership``,
    then maps the result back to the Poincaré ball.

    The computation proceeds in three steps:

    1. **Poincaré → Klein**: ``y = 2x / (1 + c‖x‖²)``.
    2. **Einstein midpoint**: ``m_K = Σ(w·γ·y) / Σ(w·γ)`` where
       ``γ = 1/√(1 − c‖y‖²)`` is the Lorentz factor.
    3. **Klein → Poincaré**: ``m_P = m_K / (1 + √(1 − c‖m_K‖²))``.

    Groups with no members (all-zero membership column) produce the
    origin, which is a valid point in the ball.

    Args:
        points: ``(K, d)`` features in the Poincaré ball.
        membership: ``(K, G)`` non-negative membership weights.
        c: Positive curvature parameter.

    Returns:
        ``(G, d)`` aggregated features in the Poincaré ball.
    """
    # 1. Poincaré → Klein
    norm_sq = jnp.sum(points ** 2, axis=-1, keepdims=True)  # (K, 1)
    klein = 2 * points / (1 + c * norm_sq)  # (K, d)

    # 2. Lorentz factor in Klein model
    k_norm_sq = jnp.sum(klein ** 2, axis=-1, keepdims=True)  # (K, 1)
    gamma = 1.0 / jnp.sqrt(jnp.maximum(1 - c * k_norm_sq, _EPS))  # (K, 1)

    # Einstein midpoint per group
    w_gamma = membership * gamma  # (K, G)
    numerator = w_gamma.T @ klein  # (G, d)
    denominator = jnp.sum(w_gamma, axis=0)[:, None]  # (G, 1)
    m_klein = numerator / jnp.maximum(denominator, _EPS)  # (G, d)

    # 3. Klein → Poincaré
    mk_norm_sq = jnp.sum(m_klein ** 2, axis=-1, keepdims=True)  # (G, 1)
    denom = 1 + jnp.sqrt(jnp.maximum(1 - c * mk_norm_sq, _EPS))
    m_poincare = m_klein / denom  # (G, d)

    return project(m_poincare, c)


# ---------------------------------------------------------------------------
# Poincaré hypergraph convolution layer
# ---------------------------------------------------------------------------


class PoincareHypergraphConv(AbstractHypergraphConv):
    """Hyperbolic hypergraph convolution in the Poincaré ball.

    Performs two-stage message passing in hyperbolic space:

    1. **Linear transform** in tangent space at the origin:
       ``log₀(x) → W·log₀(x) [+b] → exp₀(·)``.
    2. **Vertex → Hyperedge**: gyromidpoint aggregation over each
       hyperedge's member vertices.
    3. **Hyperedge → Vertex**: gyromidpoint aggregation of incident
       hyperedge representations back to each vertex.

    All intermediate and output features are guaranteed to lie strictly
    inside the Poincaré ball of radius ``1/√c``.

    Attributes:
        linear: Linear projection applied in the tangent space.
        c: Learnable curvature parameter (positive scalar).
    """

    linear: eqx.nn.Linear
    c: Float[Array, ""]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        c_init: float = 1.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize PoincareHypergraphConv.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            use_bias: Whether to include bias in the tangent-space
                linear layer.
            c_init: Initial curvature value (default 1.0, giving ball
                radius 1.0).
            key: PRNG key for weight initialisation.
        """
        self.linear = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=key)
        self.c = jnp.array(c_init, dtype=jnp.float32)

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply hyperbolic hypergraph convolution.

        Input features are projected into the Poincaré ball, linearly
        transformed in tangent space, then aggregated via gyromidpoint
        in both V→E and E→V directions.

        Args:
            hg: Input hypergraph (use ``geometry="poincare"``).

        Returns:
            Updated node features of shape ``(n, out_dim)``, all
            strictly inside the Poincaré ball of radius ``1/√c``.
        """
        c = jnp.abs(self.c) + _EPS  # ensure positive curvature
        H = hg._masked_incidence()
        x = hg.node_features

        # Ensure inputs are inside the ball
        x = project(x, c)

        # Linear transform in tangent space at the origin
        v = logmap0(x, c)
        v = jax.vmap(self.linear)(v)
        x = expmap0(v, c)
        x = project(x, c)

        # V→E: gyromidpoint aggregation
        e = gyromidpoint(x, H, c)

        # E→V: gyromidpoint aggregation
        out = gyromidpoint(e, H.T, c)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
