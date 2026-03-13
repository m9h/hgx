"""Geometric Neural ODE on Riemannian manifolds for hypergraph dynamics.

Provides ``RiemannianHypergraphODE`` that integrates node features on
Riemannian manifolds (Euclidean or Poincaré ball), keeping trajectories
on the manifold via projection steps and exponential-map-aware drift.

Requires the ``dynamics`` extra: ``pip install hgx[dynamics]``

References:
    - Ganea et al. (2018). "Hyperbolic Neural Networks." NeurIPS.
    - Lou et al. (2020). "Neural Manifold Ordinary Differential Equations."
      NeurIPS.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from hgx._conv._base import AbstractHypergraphConv
from hgx._dynamics import _hg_to_args
from hgx._hypergraph import Hypergraph


try:
    import diffrax
except ImportError as e:
    raise ImportError(
        "hgx geometric dynamics requires diffrax. "
        "Install with: pip install hgx[dynamics]"
    ) from e


# ---------------------------------------------------------------------------
# Manifold geometries
# ---------------------------------------------------------------------------


class AbstractManifold(eqx.Module):
    """Abstract Riemannian manifold for hypergraph dynamics.

    Subclasses must implement ``project``, ``expmap``, and ``logmap``.
    """

    @abc.abstractmethod
    def project(self, x: Float[Array, "n d"]) -> Float[Array, "n d"]:
        """Project points onto the manifold."""
        ...

    @abc.abstractmethod
    def expmap(
        self, x: Float[Array, "n d"], v: Float[Array, "n d"]
    ) -> Float[Array, "n d"]:
        """Riemannian exponential map at *x* applied to tangent vector *v*."""
        ...

    @abc.abstractmethod
    def logmap(
        self, x: Float[Array, "n d"], y: Float[Array, "n d"]
    ) -> Float[Array, "n d"]:
        """Riemannian logarithmic map: tangent vector at *x* pointing to *y*."""
        ...


class EuclideanManifold(AbstractManifold):
    """Euclidean (flat) geometry — trivial manifold.

    All operations reduce to standard vector arithmetic.
    """

    def project(self, x):
        return x

    def expmap(self, x, v):
        return x + v

    def logmap(self, x, y):
        return y - x


class PoincareBall(AbstractManifold):
    r"""Poincaré ball model of hyperbolic space.

    The Poincaré ball :math:`\mathbb{B}^d_c = \{x \in \mathbb{R}^d :
    c\|x\|^2 < 1\}` with constant negative curvature :math:`-c`.
    Default ``c=1.0``.

    Attributes:
        c: Positive curvature parameter (negative sectional curvature is -c).
        eps: Numerical stability constant.

    References:
        Ganea et al. (2018). "Hyperbolic Neural Networks." NeurIPS.
    """

    c: float = eqx.field(static=True, default=1.0)
    eps: float = eqx.field(static=True, default=1e-5)

    def _conformal_factor(self, x: Float[Array, "... d"]) -> Float[Array, "... 1"]:
        r"""Conformal factor :math:`\lambda_x = 2 / (1 - c\|x\|^2)`."""
        x_sqnorm = jnp.sum(x * x, axis=-1, keepdims=True)
        return 2.0 / jnp.maximum(1.0 - self.c * x_sqnorm, self.eps)

    def _mobius_add(
        self, x: Float[Array, "... d"], y: Float[Array, "... d"]
    ) -> Float[Array, "... d"]:
        r"""Möbius addition :math:`x \oplus_c y`."""
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        x_sqnorm = jnp.sum(x * x, axis=-1, keepdims=True)
        y_sqnorm = jnp.sum(y * y, axis=-1, keepdims=True)
        num = (1 + 2 * self.c * xy + self.c * y_sqnorm) * x + (
            1 - self.c * x_sqnorm
        ) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x_sqnorm * y_sqnorm
        return num / jnp.maximum(denom, self.eps)

    def project(self, x):
        max_norm = 1.0 / jnp.sqrt(self.c) - self.eps
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, self.eps)
        return jnp.where(norm > max_norm, x * max_norm / norm, x)

    def expmap(self, x, v):
        sqrt_c = jnp.sqrt(self.c)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        v_norm = jnp.maximum(v_norm, self.eps)
        lam = self._conformal_factor(x)
        t = jnp.tanh(sqrt_c * lam * v_norm / 2.0)
        direction = v / (sqrt_c * v_norm)
        return self.project(self._mobius_add(x, t * direction))

    def logmap(self, x, y):
        sqrt_c = jnp.sqrt(self.c)
        diff = self._mobius_add(-x, y)
        diff_norm = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        diff_norm = jnp.maximum(diff_norm, self.eps)
        lam = self._conformal_factor(x)
        # arctanh argument must be < 1; clip for numerical safety
        atanh_arg = jnp.minimum(sqrt_c * diff_norm, 1.0 - self.eps)
        scale = (2.0 / (sqrt_c * lam)) * jnp.arctanh(atanh_arg)
        return scale * diff / diff_norm


# ---------------------------------------------------------------------------
# Riemannian drift wrapper
# ---------------------------------------------------------------------------


class _RiemannianConvDrift(eqx.Module):
    """Wraps a hypergraph conv as a Riemannian-aware drift function.

    Projects the state onto the manifold at each drift evaluation to
    correct numerical drift, then computes the tangent vector via the
    conv layer.
    """

    conv: AbstractHypergraphConv
    activation: Callable = eqx.field(static=True)
    manifold: AbstractManifold

    def __call__(
        self,
        t: Float[Array, ""],
        y: Float[Array, "n d"],
        args: dict,
    ) -> Float[Array, "n d"]:
        y = self.manifold.project(y)
        hg = Hypergraph(
            node_features=y,
            incidence=args["incidence"],
            edge_features=args.get("edge_features"),
            positions=args.get("positions"),
            node_mask=args.get("node_mask"),
            edge_mask=args.get("edge_mask"),
            geometry=args.get("geometry"),
        )
        return self.activation(self.conv(hg))


# ---------------------------------------------------------------------------
# Riemannian Neural ODE
# ---------------------------------------------------------------------------


class RiemannianHypergraphODE(eqx.Module):
    """Neural ODE on a Riemannian manifold for hypergraph dynamics.

    Integrates node features in continuous time while keeping the
    trajectory on a specified manifold.  The conv layer computes tangent
    vectors; manifold projection inside the drift and on the output
    corrects numerical drift off the manifold.

    For ``EuclideanManifold`` this is equivalent to ``HypergraphNeuralODE``.
    For ``PoincareBall`` trajectories stay inside the unit ball.

    Attributes:
        drift: Wrapped convolution with manifold projection.
        manifold: Riemannian manifold geometry.
        solver: Diffrax ODE solver (default: Tsit5).
        stepsize_controller: Adaptive or constant step-size controller.
    """

    drift: _RiemannianConvDrift
    manifold: AbstractManifold
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        static=True
    )

    def __init__(
        self,
        conv: AbstractHypergraphConv,
        manifold: AbstractManifold | None = None,
        activation: Callable = jax.nn.tanh,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    ):
        """Initialize RiemannianHypergraphODE.

        Args:
            conv: Hypergraph convolution layer.  Must preserve feature dim.
            manifold: Riemannian manifold geometry.  Defaults to
                ``EuclideanManifold()`` (flat space).
            activation: Activation for drift (default: tanh).
            solver: Diffrax solver.  Defaults to ``Tsit5()``.
            stepsize_controller: Defaults to
                ``PIDController(rtol=1e-3, atol=1e-5)``.
        """
        if manifold is None:
            manifold = EuclideanManifold()
        self.manifold = manifold
        self.drift = _RiemannianConvDrift(
            conv=conv,
            activation=activation,
            manifold=manifold,
        )
        self.solver = solver if solver is not None else diffrax.Tsit5()
        self.stepsize_controller = (
            stepsize_controller
            if stepsize_controller is not None
            else diffrax.PIDController(rtol=1e-3, atol=1e-5)
        )

    def __call__(
        self,
        hg: Hypergraph,
        t0: float = 0.0,
        t1: float = 1.0,
        dt0: float | None = None,
        saveat: diffrax.SaveAt | None = None,
    ) -> diffrax.Solution:
        """Integrate node features from *t0* to *t1* on the manifold.

        Args:
            hg: Input hypergraph.  ``hg.node_features`` is projected onto
                the manifold and used as the initial state.
            t0: Start time.
            t1: End time.
            dt0: Initial step size.  If ``None``, the solver picks one.
            saveat: Diffrax ``SaveAt``.  Defaults to ``SaveAt(t1=True)``.

        Returns:
            Diffrax ``Solution`` with ``sol.ys`` projected onto the manifold.
        """
        if saveat is None:
            saveat = diffrax.SaveAt(t1=True)

        y0 = self.manifold.project(hg.node_features)

        term = diffrax.ODETerm(self.drift)
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=_hg_to_args(hg),
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
        )

        # Project all saved states onto the manifold
        projected_ys = jax.vmap(self.manifold.project)(sol.ys)
        return eqx.tree_at(lambda s: s.ys, sol, projected_ys)


# ---------------------------------------------------------------------------
# Convenience: trajectory extraction
# ---------------------------------------------------------------------------


def riemannian_trajectory(
    model: RiemannianHypergraphODE,
    hg: Hypergraph,
    t0: float = 0.0,
    t1: float = 1.0,
    num_steps: int = 50,
) -> tuple[Float[Array, " T"], Float[Array, "T n d"]]:
    """Integrate and return evenly-spaced trajectory snapshots on a manifold.

    Analogous to :func:`hgx.trajectory` but ensures all snapshots lie on
    the model's manifold.

    Args:
        model: A ``RiemannianHypergraphODE``.
        hg: Input hypergraph.
        t0: Start time.
        t1: End time.
        num_steps: Number of evenly spaced time points.

    Returns:
        Tuple ``(ts, features)`` where *ts* has shape ``(num_steps,)`` and
        *features* has shape ``(num_steps, n, d)``, with all features
        on the manifold.
    """
    ts = jnp.linspace(t0, t1, num_steps)
    saveat = diffrax.SaveAt(ts=ts)
    sol = model(hg, t0=t0, t1=t1, saveat=saveat)
    return ts, sol.ys
