"""Continuous-time dynamics on hypergraphs via Diffrax.

Provides Neural ODE and Neural SDE modules that evolve node features
in continuous time, using hypergraph convolution layers as the
learned vector field.

Requires the ``dynamics`` extra: ``pip install hgx[dynamics]``

References:
    - Chen et al. (2018). "Neural Ordinary Differential Equations." NeurIPS.
    - Li et al. (2020). "Scalable Gradients for Stochastic Differential
      Equations." AISTATS.
    - Kidger (2022). "On Neural Differential Equations." PhD thesis, Oxford.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Real

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


try:
    import diffrax
except ImportError as e:
    raise ImportError(
        "hgx dynamics requires diffrax. Install with: pip install hgx[dynamics]"
    ) from e


# ---------------------------------------------------------------------------
# Drift / diffusion wrappers
# ---------------------------------------------------------------------------


class _ConvDrift(eqx.Module):
    """Wraps a hypergraph conv layer as a Diffrax-compatible drift function.

    The incidence matrix and other static hypergraph structure are passed
    via ``args`` to avoid closure capture (required for Diffrax gradients).
    """

    conv: AbstractHypergraphConv
    activation: Callable = eqx.field(static=True)

    def __call__(
        self,
        t: Real[Array, ""],
        y: Float[Array, "n d"],
        args: dict,
    ) -> Float[Array, "n d"]:
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


class _FlatConvDrift(eqx.Module):
    """Wraps a hypergraph conv as drift on flattened state (n*d,).

    SDE integration in Diffrax 0.7+ requires 1D state for proper
    DiagonalLinearOperator support. This wrapper reshapes internally.
    """

    conv: AbstractHypergraphConv
    activation: Callable = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)
    node_dim: int = eqx.field(static=True)

    def __call__(
        self,
        t: Real[Array, ""],
        y: Float[Array, " nd"],
        args: dict,
    ) -> Float[Array, " nd"]:
        y2d = y.reshape(self.num_nodes, self.node_dim)
        hg = Hypergraph(
            node_features=y2d,
            incidence=args["incidence"],
            edge_features=args.get("edge_features"),
            positions=args.get("positions"),
            node_mask=args.get("node_mask"),
            edge_mask=args.get("edge_mask"),
            geometry=args.get("geometry"),
        )
        return self.activation(self.conv(hg)).reshape(-1)


class _DiagonalDiffusion(eqx.Module):
    """Learned diagonal diffusion on flattened state.

    Returns a Lineax DiagonalLinearOperator as required by Diffrax >= 0.7.0.
    Sigma is per-feature-dim and tiled across nodes.
    """

    log_sigma: Float[Array, " d"]
    num_nodes: int = eqx.field(static=True)

    def __call__(
        self,
        t: Real[Array, ""],
        y: Float[Array, " nd"],
        args: dict,
    ):
        import lineax as lx

        sigma = jnp.exp(self.log_sigma)
        diag = jnp.tile(sigma, (self.num_nodes,))
        return lx.DiagonalLinearOperator(diag)


class _CDEDrift(eqx.Module):
    """Wraps a hypergraph conv as a CDE vector field in ODE form.

    The conv maps ``(n, d) -> (n, d * control_dim)``, which is reshaped
    to ``(n, d, c)`` and contracted with ``dX/dt`` of shape ``(n, c)``
    to produce the drift ``(n, d)``.

    The control path is passed via ``args["control_path"]``.
    """

    conv: AbstractHypergraphConv
    activation: Callable = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)

    def __call__(
        self,
        t: Real[Array, ""],
        y: Float[Array, "n d"],
        args: dict,
    ) -> Float[Array, "n d"]:
        hg = Hypergraph(
            node_features=y,
            incidence=args["incidence"],
            edge_features=args.get("edge_features"),
            positions=args.get("positions"),
            node_mask=args.get("node_mask"),
            edge_mask=args.get("edge_mask"),
            geometry=args.get("geometry"),
        )
        raw = self.activation(self.conv(hg))  # (n, d*c)
        n = raw.shape[0]
        d = raw.shape[1] // self.control_dim
        matrix = raw.reshape(n, d, self.control_dim)  # (n, d, c)
        dxdt = args["control_path"].derivative(t)  # (n, c)
        return jnp.einsum("ndc,nc->nd", matrix, dxdt)


# ---------------------------------------------------------------------------
# Helper: pack/unpack Hypergraph <-> diffrax args
# ---------------------------------------------------------------------------


def _hg_to_args(hg: Hypergraph) -> dict:
    """Extract static structure from a Hypergraph for Diffrax args."""
    d: dict = {"incidence": hg.incidence}
    if hg.edge_features is not None:
        d["edge_features"] = hg.edge_features
    if hg.positions is not None:
        d["positions"] = hg.positions
    if hg.node_mask is not None:
        d["node_mask"] = hg.node_mask
    if hg.edge_mask is not None:
        d["edge_mask"] = hg.edge_mask
    if hg.geometry is not None:
        d["geometry"] = hg.geometry
    return d


# ---------------------------------------------------------------------------
# Neural ODE on hypergraphs
# ---------------------------------------------------------------------------


class HypergraphNeuralODE(eqx.Module):
    """Neural ODE where the vector field is a hypergraph convolution.

    Solves::

        dx/dt = activation(conv(Hypergraph(x(t), H)))

    where H is the static incidence matrix and x(t) are evolving node
    features.

    Attributes:
        drift: Wrapped convolution layer used as the drift function.
        solver: Diffrax ODE solver (default: Tsit5).
        stepsize_controller: Adaptive or constant step-size controller.
    """

    drift: _ConvDrift
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        static=True
    )

    def __init__(
        self,
        conv: AbstractHypergraphConv,
        activation: Callable = jax.nn.tanh,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    ):
        """Initialize HypergraphNeuralODE.

        Args:
            conv: Hypergraph convolution layer (e.g. UniGCNConv).
                Must map ``(n, in_dim) -> (n, in_dim)`` (same dimensionality).
            activation: Activation applied to conv output. Use ``jax.nn.tanh``
                (default) to bound the vector field and improve stability.
            solver: Diffrax solver. Defaults to ``Tsit5()`` (5th-order adaptive).
            stepsize_controller: Step-size controller. Defaults to
                ``PIDController(rtol=1e-3, atol=1e-5)``.
        """
        self.drift = _ConvDrift(conv=conv, activation=activation)
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
        """Integrate node features from t0 to t1.

        Args:
            hg: Input hypergraph. ``hg.node_features`` is the initial state.
            t0: Start time.
            t1: End time.
            dt0: Initial step size. If None, the solver picks one.
            saveat: Diffrax ``SaveAt`` for intermediate snapshots.
                Defaults to saving only the final state.

        Returns:
            Diffrax ``Solution`` with ``sol.ys`` containing node features
            at the requested times.
        """
        if saveat is None:
            saveat = diffrax.SaveAt(t1=True)

        term = diffrax.ODETerm(self.drift)  # pyright: ignore[reportArgumentType]
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=hg.node_features,
            args=_hg_to_args(hg),
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
        )
        return sol


# ---------------------------------------------------------------------------
# Neural SDE on hypergraphs
# ---------------------------------------------------------------------------


class HypergraphNeuralSDE(eqx.Module):
    """Neural SDE where drift is a hypergraph conv and diffusion is learned.

    Solves::

        dx = activation(conv(Hypergraph(x(t), H))) dt + sigma dW(t)

    where sigma is a learned diagonal diffusion matrix.

    Internally, the state is flattened to 1D ``(n*d,)`` for compatibility
    with Diffrax >= 0.7 ``ControlTerm`` + ``lineax.DiagonalLinearOperator``.
    The output is reshaped back to ``(n, d)`` automatically.

    This is useful for modeling stochastic developmental processes where
    cell fate decisions have an inherently noisy component.

    Attributes:
        drift: Wrapped convolution layer on flattened state.
        diffusion: Learned diagonal diffusion on flattened state.
        solver: Diffrax SDE solver (default: Euler-Maruyama).
        dt: Fixed step size for SDE integration.
        num_nodes: Number of nodes (static, for reshape).
        node_dim: Feature dimensionality (static, for reshape).
    """

    drift: _FlatConvDrift
    diffusion: _DiagonalDiffusion
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)
    node_dim: int = eqx.field(static=True)

    def __init__(
        self,
        conv: AbstractHypergraphConv,
        num_nodes: int,
        node_dim: int,
        activation: Callable = jax.nn.tanh,
        sigma_init: float = 0.1,
        solver: diffrax.AbstractSolver | None = None,
        dt: float = 0.01,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize HypergraphNeuralSDE.

        Args:
            conv: Hypergraph convolution layer. Must preserve feature dim.
            num_nodes: Number of nodes in the hypergraph.
            node_dim: Dimensionality of node features.
            activation: Activation for drift (default: tanh).
            sigma_init: Initial diffusion scale (default: 0.1).
            solver: Diffrax SDE solver. Defaults to ``Euler()``.
            dt: Fixed step size (SDEs use constant steps).
            key: PRNG key (reserved for future init).
        """
        self.drift = _FlatConvDrift(
            conv=conv,
            activation=activation,
            num_nodes=num_nodes,
            node_dim=node_dim,
        )
        self.diffusion = _DiagonalDiffusion(
            log_sigma=jnp.full((node_dim,), jnp.log(sigma_init)),
            num_nodes=num_nodes,
        )
        self.solver = solver if solver is not None else diffrax.Euler()
        self.dt = dt
        self.num_nodes = num_nodes
        self.node_dim = node_dim

    def __call__(
        self,
        hg: Hypergraph,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        key: PRNGKeyArray,
        saveat: diffrax.SaveAt | None = None,
    ) -> diffrax.Solution:
        """Integrate node features as an SDE from t0 to t1.

        Args:
            hg: Input hypergraph. ``hg.node_features`` is the initial state.
            t0: Start time.
            t1: End time.
            key: PRNG key for Brownian motion sampling.
            saveat: Diffrax ``SaveAt`` for intermediate snapshots.

        Returns:
            Diffrax ``Solution``. ``sol.ys`` has shape ``(T, n*d)``
            (flattened). Use ``sol.ys.reshape(-1, n, d)`` to recover
            the ``(T, n, d)`` node feature trajectory.
        """
        if saveat is None:
            saveat = diffrax.SaveAt(t1=True)

        nd = self.num_nodes * self.node_dim
        brownian = diffrax.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=self.dt / 2,
            shape=(nd,),
            key=key,
        )

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(self.drift),  # pyright: ignore[reportArgumentType]
            diffrax.ControlTerm(self.diffusion, brownian),  # pyright: ignore[reportArgumentType]
        )

        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=self.dt,
            y0=hg.node_features.reshape(-1),
            args=_hg_to_args(hg),
            saveat=saveat,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        return sol


# ---------------------------------------------------------------------------
# Neural CDE on hypergraphs
# ---------------------------------------------------------------------------


class HypergraphNeuralCDE(eqx.Module):
    """Neural CDE where the vector field is driven by external observations.

    Solves::

        dy/dt = f_θ(y(t)) · dX/dt

    where ``X(t)`` is a control path interpolated from irregular
    observations via cubic Hermite splines, and ``f_θ`` is a hypergraph
    convolution that outputs a matrix ``(n, d, c)`` which is contracted
    with ``dX/dt`` of shape ``(n, c)``.

    The CDE is converted to ODE form internally, so the state remains
    ``(n, d)`` without flattening.

    Attributes:
        drift: Wrapped CDE vector field (conv + reshape + contraction).
        control_dim: Dimensionality of the driving signal per node.
        solver: Diffrax ODE solver (default: Tsit5).
        stepsize_controller: Adaptive step-size controller.
    """

    drift: _CDEDrift
    control_dim: int = eqx.field(static=True)
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(
        static=True
    )

    def __init__(
        self,
        conv: AbstractHypergraphConv,
        control_dim: int,
        activation: Callable = jax.nn.tanh,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    ):
        """Initialize HypergraphNeuralCDE.

        Args:
            conv: Hypergraph convolution layer. Must map
                ``(n, in_dim) -> (n, in_dim * control_dim)`` to produce
                the matrix that is contracted with dX/dt.
            control_dim: Dimensionality of the driving signal per node.
            activation: Activation applied to conv output (default: tanh).
            solver: Diffrax solver. Defaults to ``Tsit5()``.
            stepsize_controller: Defaults to
                ``PIDController(rtol=1e-3, atol=1e-5)``.
        """
        self.drift = _CDEDrift(
            conv=conv, activation=activation, control_dim=control_dim
        )
        self.control_dim = control_dim
        self.solver = solver if solver is not None else diffrax.Tsit5()
        self.stepsize_controller = (
            stepsize_controller
            if stepsize_controller is not None
            else diffrax.PIDController(rtol=1e-3, atol=1e-5)
        )

    def __call__(
        self,
        hg: Hypergraph,
        ts: Float[Array, " T"],
        controls: Float[Array, "T n c"],
        saveat: diffrax.SaveAt | None = None,
    ) -> diffrax.Solution:
        """Integrate node features driven by a control signal.

        Args:
            hg: Input hypergraph. ``hg.node_features`` is the initial
                hidden state ``y(t0)``.
            ts: Observation times of shape ``(T,)``.
            controls: Observed signals of shape ``(T, n, c)`` at each time.
            saveat: Diffrax ``SaveAt``. Defaults to saving at all ``ts``.

        Returns:
            Diffrax ``Solution`` with ``sol.ys`` of shape ``(T, n, d)``.
        """
        if saveat is None:
            saveat = diffrax.SaveAt(ts=ts)

        coeffs = diffrax.backward_hermite_coefficients(ts, controls)
        control_path = diffrax.CubicInterpolation(ts, coeffs)

        args = _hg_to_args(hg)
        args["control_path"] = control_path

        term = diffrax.ODETerm(self.drift)  # pyright: ignore[reportArgumentType]
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=hg.node_features,
            args=args,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
        )
        return sol


# ---------------------------------------------------------------------------
# Convenience: integrate and return updated Hypergraph
# ---------------------------------------------------------------------------


def evolve(
    model: HypergraphNeuralODE | HypergraphNeuralSDE | HypergraphNeuralCDE,
    hg: Hypergraph,
    t0: float = 0.0,
    t1: float = 1.0,
    *,
    key: PRNGKeyArray | None = None,
    ts: Float[Array, " T"] | None = None,
    controls: Float[Array, "T n c"] | None = None,
) -> Hypergraph:
    """Evolve a hypergraph's node features in continuous time.

    Convenience wrapper that integrates and returns a new Hypergraph
    with updated node features at time t1.

    Args:
        model: A HypergraphNeuralODE, HypergraphNeuralSDE, or
            HypergraphNeuralCDE.
        hg: Input hypergraph.
        t0: Start time.
        t1: End time.
        key: PRNG key (required for SDE, ignored for ODE/CDE).
        ts: Observation times (required for CDE).
        controls: Control signals (required for CDE).

    Returns:
        New Hypergraph with node_features evolved to time t1.
    """
    if isinstance(model, HypergraphNeuralSDE):
        if key is None:
            raise ValueError("Must provide key for SDE integration.")
        import typing
        sol = model(hg, t0=t0, t1=t1, key=key)
        # SDE state is flattened (n*d,) -> reshape back
        ys = typing.cast(Array, sol.ys)
        new_features = ys[-1].reshape(hg.node_features.shape)
    elif isinstance(model, HypergraphNeuralCDE):
        import typing
        if ts is None or controls is None:
            raise ValueError("Must provide ts and controls for CDE integration.")
        sol = model(hg, ts=ts, controls=controls, saveat=diffrax.SaveAt(t1=True))
        ys = typing.cast(Array, sol.ys)
        new_features = ys[-1]
    else:
        import typing
        sol = model(hg, t0=t0, t1=t1)
        # ODE state is (n, d)
        ys = typing.cast(Array, sol.ys)
        new_features = ys[-1]

    return Hypergraph(
        node_features=new_features,
        incidence=hg.incidence,
        edge_features=hg.edge_features,
        positions=hg.positions,
        node_mask=hg.node_mask,
        edge_mask=hg.edge_mask,
        geometry=hg.geometry,
    )


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------


def trajectory(
    model: HypergraphNeuralODE | HypergraphNeuralSDE,
    hg: Hypergraph,
    t0: float = 0.0,
    t1: float = 1.0,
    num_steps: int = 50,
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, " T"], Float[Array, "T n d"]]:
    """Integrate and return evenly-spaced trajectory snapshots.

    Convenience function that calls the model with ``SaveAt(ts=...)``
    and reshapes the result into ``(T, n, d)`` node-feature snapshots.

    Args:
        model: A HypergraphNeuralODE or HypergraphNeuralSDE.
        hg: Input hypergraph.
        t0: Start time.
        t1: End time.
        num_steps: Number of evenly spaced time points (inclusive of t0, t1).
        key: PRNG key (required for SDE, ignored for ODE).

    Returns:
        Tuple ``(ts, features)`` where *ts* has shape ``(num_steps,)`` and
        *features* has shape ``(num_steps, n, d)``.
    """
    ts = jnp.linspace(t0, t1, num_steps)
    saveat = diffrax.SaveAt(ts=ts)

    import typing
    if isinstance(model, HypergraphNeuralSDE):
        if key is None:
            raise ValueError("Must provide key for SDE trajectory.")
        sol = model(hg, t0=t0, t1=t1, key=key, saveat=saveat)
        n, d = model.num_nodes, model.node_dim
        ys = typing.cast(Array, sol.ys)
        features = ys.reshape(num_steps, n, d)
    else:
        sol = model(hg, t0=t0, t1=t1, saveat=saveat)
        ys = typing.cast(Array, sol.ys)
        features = ys

    return ts, features
