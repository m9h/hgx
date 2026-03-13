"""Tests for the PGMax belief propagation bridge."""

import hgx
import jax
import jax.numpy as jnp
import numpy as np
import pytest


pgmax = pytest.importorskip("pgmax")

from hgx._pgmax import (
    ActiveInferenceStep,
    hypergraph_to_factor_graph,
    learn_potentials,
    run_cell_fate_inference,
)


@pytest.fixture
def simple_hg():
    """A small hypergraph with 4 nodes and 2 hyperedges."""
    return hgx.from_edge_list(
        [(0, 1, 2), (1, 2, 3)],
        num_nodes=4,
        node_features=jnp.ones((4, 3)),
    )


class TestHypergraphToFactorGraph:
    def test_returns_fg_and_variables(self, simple_hg):
        from pgmax.fgraph.fgraph import FactorGraph

        fg, variables = hypergraph_to_factor_graph(simple_hg, num_states=3)
        assert isinstance(fg, FactorGraph)
        assert variables.shape == (4,)

    def test_num_states_propagated(self, simple_hg):
        _, variables = hypergraph_to_factor_graph(simple_hg, num_states=5)
        # NDVarArray.num_states is an array; all entries should equal 5
        assert np.all(variables.num_states == 5)

    def test_custom_potential_fn(self, simple_hg):
        def pot_fn(config, members):
            # Reward agreement: all same state gets +1
            return 1.0 if len(set(config)) == 1 else 0.0

        fg, variables = hypergraph_to_factor_graph(
            simple_hg, num_states=2, potential_fn=pot_fn
        )
        assert isinstance(fg, pgmax.fgraph.FactorGraph)

    def test_skips_singleton_edges(self):
        """Hyperedges with < 2 members produce no factors."""
        hg = hgx.from_incidence(
            jnp.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
            node_features=jnp.ones((3, 2)),
        )
        # Edge 0 has 1 member, edge 1 has 2 members — should not error
        fg, variables = hypergraph_to_factor_graph(hg, num_states=2)
        assert variables.shape == (3,)


class TestLearnPotentials:
    def test_output_shape(self, simple_hg):
        key = jax.random.PRNGKey(0)
        conv = hgx.UniGCNConv(in_dim=3, out_dim=4, key=key)
        log_probs = learn_potentials(simple_hg, conv, num_states=4, key=key)
        assert log_probs.shape == (4, 4)

    def test_log_probabilities_valid(self, simple_hg):
        key = jax.random.PRNGKey(1)
        conv = hgx.UniGCNConv(in_dim=3, out_dim=3, key=key)
        log_probs = learn_potentials(simple_hg, conv, num_states=3, key=key)
        # exp(log_softmax) should sum to 1 along last axis
        probs = jnp.exp(log_probs)
        np.testing.assert_allclose(
            jnp.sum(probs, axis=-1), 1.0, atol=1e-5
        )


class TestRunCellFateInference:
    def test_returns_correct_keys(self, simple_hg):
        result = run_cell_fate_inference(simple_hg, num_fates=3, num_bp_iters=5)
        assert "marginals" in result
        assert "map_states" in result
        assert "beliefs" in result
        assert "factor_graph" in result

    def test_marginals_shape(self, simple_hg):
        result = run_cell_fate_inference(simple_hg, num_fates=3, num_bp_iters=5)
        assert result["marginals"].shape == (4, 3)

    def test_marginals_are_probabilities(self, simple_hg):
        result = run_cell_fate_inference(simple_hg, num_fates=2, num_bp_iters=20)
        row_sums = np.sum(result["marginals"], axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.1)

    def test_map_states_valid(self, simple_hg):
        num_fates = 4
        result = run_cell_fate_inference(
            simple_hg, num_fates=num_fates, num_bp_iters=10
        )
        assert result["map_states"].shape == (4,)
        assert np.all(result["map_states"] >= 0)
        assert np.all(result["map_states"] < num_fates)

    def test_with_conv_evidence(self, simple_hg):
        key = jax.random.PRNGKey(0)
        conv = hgx.UniGCNConv(in_dim=3, out_dim=3, key=key)
        result = run_cell_fate_inference(
            simple_hg, num_fates=3, conv=conv, num_bp_iters=5, key=key
        )
        assert result["marginals"].shape == (4, 3)

    def test_conv_requires_key(self, simple_hg):
        key = jax.random.PRNGKey(0)
        conv = hgx.UniGCNConv(in_dim=3, out_dim=3, key=key)
        with pytest.raises(ValueError, match="Must provide key"):
            run_cell_fate_inference(simple_hg, num_fates=3, conv=conv)


class TestActiveInferenceStep:
    """ActiveInferenceStep requires diffrax — skip if not available."""

    @pytest.fixture
    def ode_hg(self):
        """Hypergraph sized for the ODE model."""
        return hgx.from_edge_list(
            [(0, 1, 2), (1, 2, 3)],
            num_nodes=4,
            node_features=jnp.ones((4, 3)),
        )

    def test_construction(self, ode_hg):
        diffrax = pytest.importorskip("diffrax")
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key, 2)
        ode = hgx.HypergraphNeuralODE(
            hgx.UniGCNConv(in_dim=3, out_dim=3, key=k1),
        )
        conv = hgx.UniGCNConv(in_dim=3, out_dim=2, key=k2)
        step = ActiveInferenceStep(
            ode_model=ode, conv=conv, num_states=2, num_bp_iters=5
        )
        assert step.num_states == 2
        assert step.num_bp_iters == 5
