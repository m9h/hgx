"""Temporal hypergraph utilities for time-series of hypergraph snapshots.

Utilities for developmental biology trajectories (pseudotime, real
time-course data).  Supports construction, interpolation, windowing,
topology alignment, and Neural ODE fitting.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# TemporalHypergraph data structure
# ---------------------------------------------------------------------------


class TemporalHypergraph(eqx.Module):
    """A sequence of hypergraph snapshots with shared or evolving topology.

    Designed for developmental biology trajectories where node features
    evolve over time, and topology may be fixed or changing (e.g. cell
    divisions adding nodes and hyperedges).

    Attributes:
        times: Timepoints of shape ``(T,)``.
        features: Node features over time of shape ``(T, n, d)``.
        incidence: Incidence matrix — either shared ``(n, m)`` or
            per-time ``(T, n, m)``.
        edge_features: Optional per-time edge features ``(T, m, de)``.
        positions: Optional per-time spatial coordinates ``(T, n, s)``.
        node_mask: Optional per-time node masks ``(T, n)``.
            ``True`` for active nodes.
        edge_mask: Optional per-time edge masks ``(T, m)``.
            ``True`` for active hyperedges.
    """

    times: Float[Array, " T"]
    features: Float[Array, "T n d"]
    incidence: Float[Array, "n m"] | Float[Array, "T n m"]
    edge_features: Float[Array, "T m de"] | None = None
    positions: Float[Array, "T n s"] | None = None
    node_mask: Bool[Array, "T n"] | None = None
    edge_mask: Bool[Array, "T m"] | None = None

    @property
    def shared_topology(self) -> bool:
        """Whether all snapshots share the same incidence matrix."""
        return self.incidence.ndim == 2

    def __getitem__(self, t: int) -> Hypergraph:
        """Return a :class:`Hypergraph` snapshot at time index *t*."""
        inc = self.incidence if self.incidence.ndim == 2 else self.incidence[t]
        return Hypergraph(
            node_features=self.features[t],
            incidence=inc,
            edge_features=(
                None if self.edge_features is None else self.edge_features[t]
            ),
            positions=(
                None if self.positions is None else self.positions[t]
            ),
            node_mask=(
                None if self.node_mask is None else self.node_mask[t]
            ),
            edge_mask=(
                None if self.edge_mask is None else self.edge_mask[t]
            ),
        )

    def __len__(self) -> int:
        return self.times.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def from_snapshots(
    hypergraphs: list[Hypergraph],
    times: Float[Array, " T"],
) -> TemporalHypergraph:
    """Build a :class:`TemporalHypergraph` from a list of Hypergraph objects.

    All snapshots must have the same node/edge dimensions.  If incidences
    are identical across snapshots the result stores a single shared
    incidence matrix (2-D); otherwise it stores per-time incidences (3-D).

    Use :func:`align_topologies` instead if snapshots have different
    numbers of nodes or edges.

    Args:
        hypergraphs: List of :class:`Hypergraph` instances with uniform
            ``(n, d)`` features and ``(n, m)`` incidence shapes.
        times: Corresponding timepoints of shape ``(T,)``.

    Returns:
        A :class:`TemporalHypergraph`.
    """
    n0, d0 = hypergraphs[0].node_features.shape
    m0 = hypergraphs[0].incidence.shape[1]
    for i, hg in enumerate(hypergraphs):
        if hg.node_features.shape != (n0, d0):
            raise ValueError(
                f"Snapshot {i} has features shape {hg.node_features.shape}, "
                f"expected ({n0}, {d0}). Use align_topologies() for varying sizes."
            )
        if hg.incidence.shape != (n0, m0):
            raise ValueError(
                f"Snapshot {i} has incidence shape {hg.incidence.shape}, "
                f"expected ({n0}, {m0}). Use align_topologies() for varying sizes."
            )

    features = jnp.stack([hg.node_features for hg in hypergraphs])

    incidences = [hg.incidence for hg in hypergraphs]
    all_same = all(
        bool(jnp.array_equal(incidences[0], inc)) for inc in incidences[1:]
    )
    incidence = incidences[0] if all_same else jnp.stack(incidences)

    edge_features = None
    if hypergraphs[0].edge_features is not None:
        edge_features = jnp.stack([hg.edge_features for hg in hypergraphs])

    positions = None
    if hypergraphs[0].positions is not None:
        positions = jnp.stack([hg.positions for hg in hypergraphs])

    node_mask = None
    if hypergraphs[0].node_mask is not None:
        node_mask = jnp.stack([hg.node_mask for hg in hypergraphs])

    edge_mask = None
    if hypergraphs[0].edge_mask is not None:
        edge_mask = jnp.stack([hg.edge_mask for hg in hypergraphs])

    return TemporalHypergraph(
        times=jnp.asarray(times, dtype=float),
        features=features,
        incidence=incidence,
        edge_features=edge_features,
        positions=positions,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interpolate(
    temporal_hg: TemporalHypergraph,
    t: float | Float[Array, ""],
    method: str = "linear",
) -> Hypergraph:
    """Interpolate node features at arbitrary time *t* between snapshots.

    Useful for Neural ODE training on irregularly-sampled data where
    predictions and observations occur at different times.

    Args:
        temporal_hg: Source temporal hypergraph.
        t: Query time (scalar).  Clamped to the observed time range.
        method: Interpolation method (currently only ``"linear"``).

    Returns:
        A :class:`Hypergraph` with interpolated node features and the
        incidence matrix of the earlier bounding snapshot (or shared
        incidence if topology is fixed).
    """
    if method != "linear":
        raise ValueError(f"Unsupported interpolation method: {method!r}")

    times = temporal_hg.times
    T = times.shape[0]

    # Find left index: times[idx] <= t < times[idx+1]
    idx = jnp.searchsorted(times, t, side="right") - 1
    idx = jnp.clip(idx, 0, T - 2)

    dt = times[idx + 1] - times[idx]
    alpha = jnp.where(dt > 0, (t - times[idx]) / dt, 0.0)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    feat = (1.0 - alpha) * temporal_hg.features[idx] + (
        alpha * temporal_hg.features[idx + 1]
    )

    inc = (
        temporal_hg.incidence
        if temporal_hg.incidence.ndim == 2
        else temporal_hg.incidence[idx]
    )

    ef = None
    if temporal_hg.edge_features is not None:
        ef = (1.0 - alpha) * temporal_hg.edge_features[idx] + (
            alpha * temporal_hg.edge_features[idx + 1]
        )

    pos = None
    if temporal_hg.positions is not None:
        pos = (1.0 - alpha) * temporal_hg.positions[idx] + (
            alpha * temporal_hg.positions[idx + 1]
        )

    nm = None
    if temporal_hg.node_mask is not None:
        nm = temporal_hg.node_mask[idx] & temporal_hg.node_mask[idx + 1]

    em = None
    if temporal_hg.edge_mask is not None:
        em = temporal_hg.edge_mask[idx] & temporal_hg.edge_mask[idx + 1]

    return Hypergraph(
        node_features=feat,
        incidence=inc,
        edge_features=ef,
        positions=pos,
        node_mask=nm,
        edge_mask=em,
    )


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def sliding_window(
    temporal_hg: TemporalHypergraph,
    window_size: int,
    stride: int = 1,
) -> list[TemporalHypergraph]:
    """Extract overlapping sub-sequence windows for training sequence models.

    Args:
        temporal_hg: Source temporal hypergraph with *T* snapshots.
        window_size: Number of snapshots per window.
        stride: Step between successive window starts (default 1).

    Returns:
        List of :class:`TemporalHypergraph` sub-sequences, each
        containing *window_size* consecutive snapshots.
    """
    T = temporal_hg.times.shape[0]
    windows: list[TemporalHypergraph] = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        sl = slice(start, end)

        inc = (
            temporal_hg.incidence
            if temporal_hg.incidence.ndim == 2
            else temporal_hg.incidence[sl]
        )

        windows.append(
            TemporalHypergraph(
                times=temporal_hg.times[sl],
                features=temporal_hg.features[sl],
                incidence=inc,
                edge_features=(
                    None
                    if temporal_hg.edge_features is None
                    else temporal_hg.edge_features[sl]
                ),
                positions=(
                    None
                    if temporal_hg.positions is None
                    else temporal_hg.positions[sl]
                ),
                node_mask=(
                    None
                    if temporal_hg.node_mask is None
                    else temporal_hg.node_mask[sl]
                ),
                edge_mask=(
                    None
                    if temporal_hg.edge_mask is None
                    else temporal_hg.edge_mask[sl]
                ),
            )
        )

    return windows


# ---------------------------------------------------------------------------
# Topology alignment
# ---------------------------------------------------------------------------


def align_topologies(
    snapshots: list[Hypergraph],
    times: Float[Array, " T"] | None = None,
) -> TemporalHypergraph:
    """Pad snapshots with different node/edge counts to uniform shapes.

    Handles the common developmental biology scenario where cells divide
    (adding nodes) and new interactions form (adding hyperedges) over
    time.  All snapshots are zero-padded to the maximum observed size and
    boolean masks mark the active nodes and edges at each time step.

    The result has per-time incidence (3-D) and is suitable for
    ``jax.vmap`` over the time axis.

    Args:
        snapshots: List of :class:`Hypergraph` instances, possibly with
            different numbers of nodes and/or edges.
        times: Timepoints of shape ``(T,)``.  Defaults to
            ``[0, 1, ..., T-1]``.

    Returns:
        A :class:`TemporalHypergraph` with uniform shapes and masks.
    """
    T = len(snapshots)
    if times is None:
        times = jnp.arange(T, dtype=float)
    else:
        times = jnp.asarray(times, dtype=float)

    max_n = max(hg.node_features.shape[0] for hg in snapshots)
    max_m = max(hg.incidence.shape[1] for hg in snapshots)

    padded_features: list = []
    padded_incidences: list = []
    node_masks: list = []
    edge_masks: list = []
    padded_ef: list = []
    padded_pos: list = []

    has_ef = snapshots[0].edge_features is not None
    has_pos = snapshots[0].positions is not None

    for hg in snapshots:
        n = hg.node_features.shape[0]
        m = hg.incidence.shape[1]

        padded_features.append(
            jnp.pad(hg.node_features, ((0, max_n - n), (0, 0)))
        )
        padded_incidences.append(
            jnp.pad(hg.incidence, ((0, max_n - n), (0, max_m - m)))
        )

        node_masks.append(
            jnp.concatenate([
                jnp.ones(n, dtype=bool),
                jnp.zeros(max_n - n, dtype=bool),
            ])
        )
        edge_masks.append(
            jnp.concatenate([
                jnp.ones(m, dtype=bool),
                jnp.zeros(max_m - m, dtype=bool),
            ])
        )

        if has_ef:
            padded_ef.append(
                jnp.pad(hg.edge_features, ((0, max_m - m), (0, 0)))
            )
        if has_pos:
            padded_pos.append(
                jnp.pad(hg.positions, ((0, max_n - n), (0, 0)))
            )

    return TemporalHypergraph(
        times=times,
        features=jnp.stack(padded_features),
        incidence=jnp.stack(padded_incidences),
        edge_features=jnp.stack(padded_ef) if has_ef else None,
        positions=jnp.stack(padded_pos) if has_pos else None,
        node_mask=jnp.stack(node_masks),
        edge_mask=jnp.stack(edge_masks),
    )


# ---------------------------------------------------------------------------
# Temporal smoothness regularization
# ---------------------------------------------------------------------------


def temporal_smoothness_loss(
    temporal_hg: TemporalHypergraph,
) -> Float[Array, ""]:
    r"""Regularization loss encouraging smooth feature evolution.

    Computes :math:`\sum_t \|x(t{+}1) - x(t)\|^2` over consecutive
    snapshots.  If node masks are present, masked-out nodes do not
    contribute to the loss.

    Args:
        temporal_hg: Temporal hypergraph with at least two snapshots.

    Returns:
        Scalar loss value.
    """
    diffs = temporal_hg.features[1:] - temporal_hg.features[:-1]
    if temporal_hg.node_mask is not None:
        mask = temporal_hg.node_mask[:-1] & temporal_hg.node_mask[1:]
        diffs = diffs * mask[..., None]
    return jnp.sum(diffs**2)


# ---------------------------------------------------------------------------
# Neural ODE fitting convenience
# ---------------------------------------------------------------------------


def fit_neural_ode(
    temporal_hg: TemporalHypergraph,
    conv: AbstractHypergraphConv,
    *,
    key: PRNGKeyArray,
    epochs: int = 100,
    lr: float = 1e-3,
):
    """Fit a :class:`~hgx.HypergraphNeuralODE` to a temporal trajectory.

    Uses the first snapshot as the initial condition and trains to
    minimise MSE at all subsequent timepoints.

    Requires ``diffrax`` and ``optax``::

        pip install hgx[dynamics] optax

    Args:
        temporal_hg: Observed trajectory.
        conv: Hypergraph convolution layer (in_dim == out_dim).
        key: PRNG key (reserved for future use).
        epochs: Number of training epochs (default 100).
        lr: Learning rate for Adam (default 1e-3).

    Returns:
        Trained :class:`~hgx.HypergraphNeuralODE`.
    """
    try:
        import diffrax
    except ImportError as e:
        raise ImportError(
            "fit_neural_ode requires diffrax: pip install hgx[dynamics]"
        ) from e
    try:
        import optax
    except ImportError as e:
        raise ImportError(
            "fit_neural_ode requires optax: pip install optax"
        ) from e

    from hgx._dynamics import HypergraphNeuralODE

    model = HypergraphNeuralODE(conv)
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    hg0 = temporal_hg[0]
    target_times = temporal_hg.times[1:]
    targets = temporal_hg.features[1:]

    t0 = float(temporal_hg.times[0])
    t1 = float(temporal_hg.times[-1])
    has_mask = temporal_hg.node_mask is not None
    target_mask = temporal_hg.node_mask[1:] if has_mask else None

    @eqx.filter_jit
    def train_step(model, opt_state):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(
                hg0,
                t0=t0,
                t1=t1,
                saveat=diffrax.SaveAt(ts=target_times),
            )
            diff = (sol.ys - targets) ** 2
            if has_mask:
                diff = diff * target_mask[..., None]
            return jnp.mean(diff)

        loss, grads = loss_fn(model)
        updates, new_opt_state = optim.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    for _ in range(epochs):
        model, opt_state, _loss = train_step(model, opt_state)

    return model
