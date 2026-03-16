"""Tests for Neural Developmental Programs on hypergraphs."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._ndp import CellProgram, develop_trajectory, HypergraphNDP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_hypergraph():
    """Small hypergraph with 4 nodes, 2 edges, feature dim 8."""
    H = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    features = jnp.ones((4, 8))
    return hgx.from_incidence(H, node_features=features)


@pytest.fixture
def cell_program(prng_key):
    """CellProgram with state_dim=8, hidden_dim=16."""
    return CellProgram(state_dim=8, hidden_dim=16, key=prng_key)


@pytest.fixture
def ndp_model(prng_key):
    """HypergraphNDP with max_nodes=12, max_edges=6."""
    k1, k2 = jax.random.split(prng_key)
    program = CellProgram(state_dim=8, hidden_dim=16, key=k1)
    conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k2)
    return HypergraphNDP(
        program=program,
        conv=conv,
        max_nodes=12,
        max_edges=6,
    )


# ---------------------------------------------------------------------------
# CellProgram tests
# ---------------------------------------------------------------------------


class TestCellProgram:
    """Test the shared CellProgram (DNA) module."""

    def test_forward_pass_shapes(self, cell_program):
        """Output shapes should match specification."""
        state = jnp.ones(8)
        neighbor = jnp.zeros(8)
        state_update, grow_logit, connect_logits = cell_program(state, neighbor)

        assert state_update.shape == (8,)
        assert grow_logit.shape == ()
        assert connect_logits.shape == (16,)

    def test_output_is_finite(self, cell_program):
        """All outputs should be finite."""
        state = jnp.ones(8)
        neighbor = jnp.ones(8) * 0.5
        state_update, grow_logit, connect_logits = cell_program(state, neighbor)

        assert jnp.all(jnp.isfinite(state_update))
        assert jnp.isfinite(grow_logit)
        assert jnp.all(jnp.isfinite(connect_logits))

    def test_vmap_over_nodes(self, cell_program):
        """CellProgram should vmap cleanly over a batch of nodes."""
        n = 6
        states = jnp.ones((n, 8))
        neighbors = jnp.zeros((n, 8))

        updates, logits, connects = jax.vmap(cell_program)(states, neighbors)

        assert updates.shape == (n, 8)
        assert logits.shape == (n,)
        assert connects.shape == (n, 16)

    def test_different_inputs_give_different_outputs(self, cell_program):
        """Different local states should produce different outputs."""
        state_a = jnp.ones(8)
        state_b = jnp.zeros(8)
        neighbor = jnp.ones(8) * 0.5

        update_a, _, _ = cell_program(state_a, neighbor)
        update_b, _, _ = cell_program(state_b, neighbor)

        assert not jnp.allclose(update_a, update_b)


# ---------------------------------------------------------------------------
# HypergraphNDP single-step tests
# ---------------------------------------------------------------------------


class TestNDPSingleStep:
    """Test a single developmental step."""

    def test_output_is_hypergraph(self, ndp_model, small_hypergraph, prng_key):
        """Single step should return a Hypergraph."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)
        result = ndp_model(hg, key=prng_key)
        assert isinstance(result, hgx.Hypergraph)

    def test_output_shapes_preserved(self, ndp_model, small_hypergraph, prng_key):
        """Array shapes should match the pre-allocated capacity."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)
        result = ndp_model(hg, key=prng_key)

        assert result.node_features.shape == (12, 8)
        assert result.incidence.shape == (12, 6)
        assert result.node_mask.shape == (12,)
        assert result.edge_mask.shape == (6,)

    def test_features_change_after_step(self, ndp_model, small_hypergraph, prng_key):
        """Active node features should change after a step."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)
        result = ndp_model(hg, key=prng_key)

        # At least some active node features should differ
        orig = hg.node_features[:4]
        updated = result.node_features[:4]
        assert not jnp.allclose(orig, updated)

    def test_inactive_features_unchanged_if_no_growth(self, small_hypergraph, prng_key):
        """With a very negative grow threshold, inactive slots stay zero."""
        k1, k2 = jax.random.split(prng_key)
        program = CellProgram(state_dim=8, hidden_dim=16, key=k1)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k2)
        # Very high threshold => no growth
        ndp = HypergraphNDP(
            program=program,
            conv=conv,
            max_nodes=12,
            max_edges=6,
            growth_threshold=100.0,
        )
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)
        result = ndp(hg, key=prng_key)

        # No new nodes should be activated
        assert int(jnp.sum(result.node_mask)) == 4

    def test_all_outputs_finite(self, ndp_model, small_hypergraph, prng_key):
        """All node features should be finite after a step."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)
        result = ndp_model(hg, key=prng_key)
        assert jnp.all(jnp.isfinite(result.node_features))


# ---------------------------------------------------------------------------
# Multi-step development (scan) tests
# ---------------------------------------------------------------------------


class TestNDPDevelopment:
    """Test multi-step development via jax.lax.scan."""

    def test_develop_returns_hypergraph(self, ndp_model, small_hypergraph, prng_key):
        """develop() should return a Hypergraph."""
        result = ndp_model.develop(small_hypergraph, num_steps=3, key=prng_key)
        assert isinstance(result, hgx.Hypergraph)

    def test_develop_shapes(self, ndp_model, small_hypergraph, prng_key):
        """Shapes after multi-step development should be consistent."""
        result = ndp_model.develop(small_hypergraph, num_steps=5, key=prng_key)
        assert result.node_features.shape == (12, 8)
        assert result.incidence.shape == (12, 6)

    def test_develop_auto_preallocates(self, ndp_model, small_hypergraph, prng_key):
        """develop() should auto-preallocate if shapes do not match."""
        # small_hypergraph has shape (4, 2), not (12, 6)
        result = ndp_model.develop(small_hypergraph, num_steps=3, key=prng_key)
        assert result.node_features.shape == (12, 8)

    def test_develop_features_evolve(self, ndp_model, small_hypergraph, prng_key):
        """Features should change over multiple steps."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)
        result = ndp_model.develop(hg, num_steps=5, key=prng_key)

        # Active features should have evolved
        assert not jnp.allclose(result.node_features[:4], hg.node_features[:4])


# ---------------------------------------------------------------------------
# Topology growth tests
# ---------------------------------------------------------------------------


class TestTopologyGrowth:
    """Test that masked nodes can become active during development."""

    def test_growth_with_low_threshold(self, small_hypergraph, prng_key):
        """With threshold=0, all nodes try to grow => slots fill up."""
        k1, k2 = jax.random.split(prng_key)
        program = CellProgram(state_dim=8, hidden_dim=16, key=k1)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k2)
        ndp = HypergraphNDP(
            program=program,
            conv=conv,
            max_nodes=8,
            max_edges=6,
            growth_threshold=0.0,  # everything grows
        )
        hg = hgx.preallocate(small_hypergraph, max_nodes=8, max_edges=6)
        initial_active = int(jnp.sum(hg.node_mask))

        result = ndp(hg, key=prng_key)
        final_active = int(jnp.sum(result.node_mask))

        # With threshold=0, sigmoid(any logit) > 0, so growth should happen
        assert final_active > initial_active

    def test_growth_preserves_structure(self, small_hypergraph, prng_key):
        """Original active nodes should still be active after growth."""
        k1, k2 = jax.random.split(prng_key)
        program = CellProgram(state_dim=8, hidden_dim=16, key=k1)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k2)
        ndp = HypergraphNDP(
            program=program,
            conv=conv,
            max_nodes=10,
            max_edges=6,
            growth_threshold=0.0,
        )
        hg = hgx.preallocate(small_hypergraph, max_nodes=10, max_edges=6)
        result = ndp(hg, key=prng_key)

        # Original 4 nodes should still be active
        assert jnp.all(result.node_mask[:4])

    def test_no_growth_beyond_capacity(self, small_hypergraph, prng_key):
        """Cannot grow beyond max_nodes even with threshold=0."""
        k1, k2 = jax.random.split(prng_key)
        program = CellProgram(state_dim=8, hidden_dim=16, key=k1)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k2)
        ndp = HypergraphNDP(
            program=program,
            conv=conv,
            max_nodes=6,
            max_edges=6,
            growth_threshold=0.0,
        )
        hg = hgx.preallocate(small_hypergraph, max_nodes=6, max_edges=6)

        # Run many steps to saturate
        result = ndp.develop(hg, num_steps=10, key=prng_key)
        assert int(jnp.sum(result.node_mask)) <= 6

    def test_multi_step_growth(self, small_hypergraph, prng_key):
        """Nodes should accumulate over multiple steps."""
        k1, k2 = jax.random.split(prng_key)
        program = CellProgram(state_dim=8, hidden_dim=16, key=k1)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k2)
        ndp = HypergraphNDP(
            program=program,
            conv=conv,
            max_nodes=20,
            max_edges=6,
            growth_threshold=0.0,
        )
        hg = hgx.preallocate(small_hypergraph, max_nodes=20, max_edges=6)

        result = ndp.develop(hg, num_steps=3, key=prng_key)
        final_active = int(jnp.sum(result.node_mask))

        # Should have grown beyond the initial 4 over 3 steps
        assert final_active > 4


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """All NDP operations must work under eqx.filter_jit."""

    def test_cell_program_jit(self, cell_program):
        """CellProgram should work under JIT."""

        @eqx.filter_jit
        def run(prog, s, n):
            return prog(s, n)

        state = jnp.ones(8)
        neighbor = jnp.zeros(8)
        update, logit, connects = run(cell_program, state, neighbor)

        assert update.shape == (8,)
        assert jnp.isfinite(logit)

    def test_single_step_jit(self, ndp_model, small_hypergraph, prng_key):
        """Single NDP step should work under JIT."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)

        @eqx.filter_jit
        def step(model, hg, key):
            return model(hg, key=key)

        result = step(ndp_model, hg, prng_key)
        assert isinstance(result, hgx.Hypergraph)
        assert result.node_features.shape == (12, 8)
        assert jnp.all(jnp.isfinite(result.node_features))

    def test_develop_jit(self, ndp_model, small_hypergraph, prng_key):
        """Multi-step develop should work under JIT."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)

        @eqx.filter_jit
        def run_develop(model, hg, key):
            return model.develop(hg, num_steps=3, key=key)

        result = run_develop(ndp_model, hg, prng_key)
        assert result.node_features.shape == (12, 8)
        assert jnp.all(jnp.isfinite(result.node_features))

    def test_develop_trajectory_jit(self, ndp_model, small_hypergraph, prng_key):
        """develop_trajectory should work under JIT."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)

        @eqx.filter_jit
        def run_traj(model, hg, key):
            return develop_trajectory(model, hg, num_steps=4, key=key)

        ts, features = run_traj(ndp_model, hg, prng_key)
        assert ts.shape == (4,)
        assert features.shape == (4, 12, 8)
        assert jnp.all(jnp.isfinite(features))


# ---------------------------------------------------------------------------
# develop_trajectory tests
# ---------------------------------------------------------------------------


class TestDevelopTrajectory:
    """Test the develop_trajectory functional helper."""

    def test_trajectory_shapes(self, ndp_model, small_hypergraph, prng_key):
        """Output shapes should be (num_steps,) and (num_steps, n, d)."""
        ts, features = develop_trajectory(
            ndp_model, small_hypergraph, num_steps=5, key=prng_key
        )
        assert ts.shape == (5,)
        assert features.shape == (5, 12, 8)

    def test_trajectory_ts_values(self, ndp_model, small_hypergraph, prng_key):
        """Time steps should be 0, 1, 2, ..., num_steps-1."""
        ts, _ = develop_trajectory(
            ndp_model, small_hypergraph, num_steps=4, key=prng_key
        )
        expected = jnp.array([0.0, 1.0, 2.0, 3.0])
        assert jnp.allclose(ts, expected)

    def test_trajectory_features_evolve(self, ndp_model, small_hypergraph, prng_key):
        """Features should differ between first and last step."""
        _, features = develop_trajectory(
            ndp_model, small_hypergraph, num_steps=5, key=prng_key
        )
        assert not jnp.allclose(features[0], features[-1])

    def test_trajectory_all_finite(self, ndp_model, small_hypergraph, prng_key):
        """All trajectory features should be finite."""
        _, features = develop_trajectory(
            ndp_model, small_hypergraph, num_steps=5, key=prng_key
        )
        assert jnp.all(jnp.isfinite(features))


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradients:
    """Test that gradients flow through the developmental program."""

    def test_grad_through_single_step(self, ndp_model, small_hypergraph, prng_key):
        """Gradients should flow through a single NDP step."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)

        @eqx.filter_value_and_grad
        def loss_fn(model):
            result = model(hg, key=prng_key)
            return jnp.sum(result.node_features[:4])

        loss, grads = loss_fn(ndp_model)
        assert jnp.isfinite(loss)
        # Conv weights should receive gradients
        assert not jnp.allclose(grads.conv.linear.weight, 0.0)

    def test_grad_through_develop(self, ndp_model, small_hypergraph, prng_key):
        """Gradients should flow through multi-step development."""
        hg = hgx.preallocate(small_hypergraph, max_nodes=12, max_edges=6)

        @eqx.filter_value_and_grad
        def loss_fn(model):
            result = model.develop(hg, num_steps=3, key=prng_key)
            return jnp.sum(result.node_features[:4])

        loss, grads = loss_fn(ndp_model)
        assert jnp.isfinite(loss)
        # MLP weights in the program should receive gradients
        assert not jnp.allclose(grads.program.state_mlp.layers[0].weight, 0.0)
