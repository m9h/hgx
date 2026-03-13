"""Tests for temporal hypergraph utilities."""

import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._temporal import (
    align_topologies,
    fit_neural_ode,
    from_snapshots,
    interpolate,
    sliding_window,
    temporal_smoothness_loss,
    TemporalHypergraph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def shared_snapshots():
    """Three snapshots with shared topology and linear feature growth."""
    H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    hgs = []
    for t in range(3):
        feat = jnp.ones((3, 4)) * (t + 1.0)
        hgs.append(hgx.from_incidence(H, node_features=feat))
    return hgs


@pytest.fixture
def varying_snapshots():
    """Three snapshots with different incidences."""
    snaps = []
    for t in range(3):
        H = jnp.eye(3).at[0, 1].set(float(t))
        feat = jnp.ones((3, 4)) * (t + 1.0)
        snaps.append(hgx.from_incidence(H, node_features=feat))
    return snaps


@pytest.fixture
def times_3():
    return jnp.array([0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# TemporalHypergraph construction
# ---------------------------------------------------------------------------


class TestTemporalHypergraph:
    def test_construction(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        assert thg.times.shape == (3,)
        assert thg.features.shape == (3, 3, 4)

    def test_shared_topology(self, shared_snapshots, times_3):
        """Identical incidences produce 2-D (shared) incidence."""
        thg = from_snapshots(shared_snapshots, times_3)
        assert thg.incidence.ndim == 2
        assert thg.shared_topology

    def test_varying_topology(self, varying_snapshots, times_3):
        """Different incidences produce 3-D (per-time) incidence."""
        thg = from_snapshots(varying_snapshots, times_3)
        assert thg.incidence.ndim == 3
        assert not thg.shared_topology

    def test_mismatched_shapes_error(self, times_3):
        """Mismatched feature shapes raise ValueError."""
        hg_a = hgx.from_incidence(jnp.eye(3), node_features=jnp.ones((3, 4)))
        hg_b = hgx.from_incidence(jnp.eye(4), node_features=jnp.ones((4, 4)))
        with pytest.raises(ValueError, match="align_topologies"):
            from_snapshots([hg_a, hg_b], times_3[:2])


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_getitem_returns_hypergraph(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        snap = thg[1]
        assert isinstance(snap, hgx.Hypergraph)
        assert snap.node_features.shape == (3, 4)
        assert jnp.allclose(snap.node_features, 2.0)

    def test_getitem_shared_incidence(self, shared_snapshots, times_3):
        """Shared topology: all snapshots get the same incidence."""
        thg = from_snapshots(shared_snapshots, times_3)
        assert jnp.array_equal(thg[0].incidence, thg[2].incidence)

    def test_getitem_varying_incidence(self, varying_snapshots, times_3):
        """Per-time topology: each snapshot gets its own incidence."""
        thg = from_snapshots(varying_snapshots, times_3)
        assert not jnp.array_equal(thg[0].incidence, thg[1].incidence)

    def test_len(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        assert len(thg) == 3

    def test_iter(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        snaps = list(thg)
        assert len(snaps) == 3
        assert all(isinstance(s, hgx.Hypergraph) for s in snaps)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


class TestInterpolate:
    def test_at_exact_time(self, shared_snapshots, times_3):
        """Interpolation at an observed time returns that snapshot's features."""
        thg = from_snapshots(shared_snapshots, times_3)
        hg = interpolate(thg, 0.5)
        # t=0.5 is the second snapshot (features = 2.0)
        assert jnp.allclose(hg.node_features, 2.0, atol=1e-5)

    def test_midpoint(self, shared_snapshots, times_3):
        """Midpoint between t=0.0 (feat=1) and t=0.5 (feat=2) is 1.5."""
        thg = from_snapshots(shared_snapshots, times_3)
        hg = interpolate(thg, 0.25)
        assert jnp.allclose(hg.node_features, 1.5, atol=1e-5)

    def test_returns_hypergraph(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        hg = interpolate(thg, 0.7)
        assert isinstance(hg, hgx.Hypergraph)
        assert hg.incidence.shape == (3, 2)

    def test_clamp_below(self, shared_snapshots, times_3):
        """Time below range clamps to first interval."""
        thg = from_snapshots(shared_snapshots, times_3)
        hg = interpolate(thg, -1.0)
        # Clamped alpha=0 -> first snapshot features
        assert jnp.allclose(hg.node_features, 1.0, atol=1e-5)

    def test_clamp_above(self, shared_snapshots, times_3):
        """Time above range clamps to last interval."""
        thg = from_snapshots(shared_snapshots, times_3)
        hg = interpolate(thg, 5.0)
        # Clamped alpha=1 -> last snapshot features
        assert jnp.allclose(hg.node_features, 3.0, atol=1e-5)

    def test_unsupported_method(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        with pytest.raises(ValueError, match="Unsupported"):
            interpolate(thg, 0.5, method="cubic")


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_count(self, shared_snapshots, times_3):
        """5 snapshots, window=3, stride=1 -> 3 windows."""
        # Build 5 snapshots
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        snaps = [
            hgx.from_incidence(H, node_features=jnp.ones((3, 4)) * i)
            for i in range(5)
        ]
        thg = from_snapshots(snaps, jnp.arange(5, dtype=float))
        windows = sliding_window(thg, window_size=3, stride=1)
        assert len(windows) == 3

    def test_window_size(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        windows = sliding_window(thg, window_size=2)
        assert all(len(w) == 2 for w in windows)

    def test_stride(self):
        """window=2, stride=2 on 4 snapshots -> 2 non-overlapping windows."""
        H = jnp.eye(3)
        snaps = [
            hgx.from_incidence(H, node_features=jnp.ones((3, 2)) * i)
            for i in range(4)
        ]
        thg = from_snapshots(snaps, jnp.arange(4, dtype=float))
        windows = sliding_window(thg, window_size=2, stride=2)
        assert len(windows) == 2
        # First window starts at t=0, second at t=2
        assert float(windows[0].times[0]) == 0.0
        assert float(windows[1].times[0]) == 2.0

    def test_returns_temporal_hypergraphs(self, shared_snapshots, times_3):
        thg = from_snapshots(shared_snapshots, times_3)
        windows = sliding_window(thg, window_size=2)
        assert all(isinstance(w, TemporalHypergraph) for w in windows)

    def test_shared_topology_preserved(self, shared_snapshots, times_3):
        """Windows of shared-topology THG also have shared (2-D) incidence."""
        thg = from_snapshots(shared_snapshots, times_3)
        windows = sliding_window(thg, window_size=2)
        assert all(w.incidence.ndim == 2 for w in windows)


# ---------------------------------------------------------------------------
# Topology alignment
# ---------------------------------------------------------------------------


class TestAlignTopologies:
    def test_uniform_padding(self):
        """Snapshots with different sizes get padded to max."""
        hg_small = hgx.from_incidence(
            jnp.eye(2), node_features=jnp.ones((2, 4))
        )
        hg_large = hgx.from_incidence(
            jnp.eye(4), node_features=jnp.ones((4, 4))
        )
        thg = align_topologies([hg_small, hg_large])
        assert thg.features.shape == (2, 4, 4)
        assert thg.incidence.shape == (2, 4, 4)

    def test_masks_present(self):
        """Alignment produces node and edge masks."""
        hg_small = hgx.from_incidence(
            jnp.eye(2), node_features=jnp.ones((2, 4))
        )
        hg_large = hgx.from_incidence(
            jnp.eye(4), node_features=jnp.ones((4, 4))
        )
        thg = align_topologies([hg_small, hg_large])
        assert thg.node_mask is not None
        assert thg.edge_mask is not None

    def test_mask_values(self):
        """Masks correctly mark active vs padded nodes/edges."""
        hg_2 = hgx.from_incidence(
            jnp.ones((2, 1)), node_features=jnp.ones((2, 3))
        )
        hg_3 = hgx.from_incidence(
            jnp.ones((3, 2)), node_features=jnp.ones((3, 3))
        )
        thg = align_topologies([hg_2, hg_3])

        # First snapshot: 2 active nodes, 1 active edge
        assert jnp.sum(thg.node_mask[0]) == 2
        assert jnp.sum(thg.edge_mask[0]) == 1
        # Second snapshot: 3 active nodes, 2 active edges
        assert jnp.sum(thg.node_mask[1]) == 3
        assert jnp.sum(thg.edge_mask[1]) == 2

    def test_masked_growth(self):
        """Simulated cell division: 2 cells -> 3 cells -> 4 cells."""
        snaps = []
        for n_cells in [2, 3, 4]:
            H = jnp.ones((n_cells, 1))  # single hyperedge
            feat = jnp.arange(n_cells * 2, dtype=float).reshape(n_cells, 2)
            snaps.append(hgx.from_incidence(H, node_features=feat))

        times = jnp.array([0.0, 1.0, 2.0])
        thg = align_topologies(snaps, times)

        # All padded to max: 4 nodes, 1 edge
        assert thg.features.shape == (3, 4, 2)
        assert thg.incidence.shape == (3, 4, 1)

        # Active node counts grow
        assert int(jnp.sum(thg.node_mask[0])) == 2
        assert int(jnp.sum(thg.node_mask[1])) == 3
        assert int(jnp.sum(thg.node_mask[2])) == 4

        # Padded features are zero
        assert float(thg.features[0, 3, 0]) == 0.0

    def test_default_times(self):
        """Omitting times produces 0, 1, ..., T-1."""
        hg = hgx.from_incidence(jnp.eye(2), node_features=jnp.ones((2, 3)))
        thg = align_topologies([hg, hg])
        assert jnp.allclose(thg.times, jnp.array([0.0, 1.0]))

    def test_per_time_incidence(self):
        """Aligned result always has per-time (3-D) incidence."""
        hg = hgx.from_incidence(jnp.eye(2), node_features=jnp.ones((2, 3)))
        thg = align_topologies([hg, hg])
        assert thg.incidence.ndim == 3


# ---------------------------------------------------------------------------
# Temporal smoothness loss
# ---------------------------------------------------------------------------


class TestTemporalSmoothnessLoss:
    def test_zero_for_constant(self, shared_snapshots, times_3):
        """Constant features across time give zero loss."""
        H = jnp.eye(3)
        snaps = [
            hgx.from_incidence(H, node_features=jnp.ones((3, 4)))
            for _ in range(3)
        ]
        thg = from_snapshots(snaps, times_3)
        loss = temporal_smoothness_loss(thg)
        assert float(loss) == 0.0

    def test_known_value(self):
        """Compute expected loss for known feature trajectory."""
        # Two snapshots: features 0 and 1
        H = jnp.eye(2)
        hg0 = hgx.from_incidence(H, node_features=jnp.zeros((2, 3)))
        hg1 = hgx.from_incidence(H, node_features=jnp.ones((2, 3)))
        thg = from_snapshots([hg0, hg1], jnp.array([0.0, 1.0]))
        loss = temporal_smoothness_loss(thg)
        # diff = ones(2,3), sum of squares = 2*3 = 6
        assert float(loss) == pytest.approx(6.0)

    def test_masked_nodes_excluded(self):
        """Masked-out nodes don't contribute to loss."""
        snaps = []
        for n in [2, 3]:
            H = jnp.ones((n, 1))
            feat = jnp.ones((n, 2)) * float(n)
            snaps.append(hgx.from_incidence(H, node_features=feat))

        thg = align_topologies(snaps, jnp.array([0.0, 1.0]))
        loss = temporal_smoothness_loss(thg)

        # Only first 2 nodes active at both times.
        # diff for active nodes: (3-2)=1, so 2 nodes * 2 features = 4
        assert float(loss) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# fit_neural_ode
# ---------------------------------------------------------------------------


class TestFitNeuralODE:
    def test_basic_fit(self, prng_key):
        """fit_neural_ode runs and returns a HypergraphNeuralODE."""
        from hgx._dynamics import HypergraphNeuralODE

        H = jnp.eye(3)
        snaps = [
            hgx.from_incidence(H, node_features=jnp.ones((3, 4)) * t)
            for t in range(3)
        ]
        thg = from_snapshots(snaps, jnp.array([0.0, 0.5, 1.0]))

        conv = hgx.UniGCNConv(in_dim=4, out_dim=4, key=prng_key)
        model = fit_neural_ode(thg, conv, key=prng_key, epochs=3, lr=1e-3)

        assert isinstance(model, HypergraphNeuralODE)

    def test_loss_decreases(self, prng_key):
        """Loss after training is lower than before."""
        import diffrax

        H = jnp.eye(3)
        snaps = [
            hgx.from_incidence(H, node_features=jnp.ones((3, 4)) * t)
            for t in range(3)
        ]
        thg = from_snapshots(snaps, jnp.array([0.0, 0.5, 1.0]))
        targets = thg.features[1:]

        conv = hgx.UniGCNConv(in_dim=4, out_dim=4, key=prng_key)

        # Loss before training
        from hgx._dynamics import HypergraphNeuralODE

        model_before = HypergraphNeuralODE(conv)
        sol = model_before(
            thg[0], t0=0.0, t1=1.0,
            saveat=diffrax.SaveAt(ts=thg.times[1:]),
        )
        loss_before = float(jnp.mean((sol.ys - targets) ** 2))

        # Train
        model_after = fit_neural_ode(
            thg, conv, key=prng_key, epochs=30, lr=1e-2
        )
        sol = model_after(
            thg[0], t0=0.0, t1=1.0,
            saveat=diffrax.SaveAt(ts=thg.times[1:]),
        )
        loss_after = float(jnp.mean((sol.ys - targets) ** 2))

        assert loss_after < loss_before
