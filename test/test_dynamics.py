"""Tests for continuous-time dynamics on hypergraphs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import hgx
from hgx._dynamics import (
    HypergraphNeuralCDE,
    HypergraphNeuralODE,
    HypergraphNeuralSDE,
    evolve,
    trajectory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def square_tiny_hypergraph():
    """Tiny hypergraph with square feature dim (in=out for conv)."""
    H = jnp.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    features = jnp.ones((4, 8))
    return hgx.from_incidence(H, node_features=features)


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Neural ODE tests
# ---------------------------------------------------------------------------


class TestHypergraphNeuralODE:
    """Test Neural ODE on hypergraphs."""

    def test_output_shape(self, square_tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)
        sol = model(square_tiny_hypergraph, t0=0.0, t1=1.0)
        # SaveAt(t1=True) -> sol.ys has shape (1, n, d)
        assert sol.ys.shape == (1, 4, 8)

    def test_features_change(self, square_tiny_hypergraph, prng_key):
        """Integration should change features (non-trivial dynamics)."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)
        sol = model(square_tiny_hypergraph, t0=0.0, t1=1.0)
        y_final = sol.ys[-1]
        assert not jnp.allclose(y_final, square_tiny_hypergraph.node_features)

    def test_short_integration_close_to_initial(
        self, square_tiny_hypergraph, prng_key
    ):
        """Very short integration should stay close to initial state."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)
        sol = model(square_tiny_hypergraph, t0=0.0, t1=1e-6)
        y_final = sol.ys[-1]
        assert jnp.allclose(
            y_final, square_tiny_hypergraph.node_features, atol=1e-3
        )

    def test_saveat_multiple_times(self, square_tiny_hypergraph, prng_key):
        """SaveAt with multiple time points returns a trajectory."""
        import diffrax

        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)
        ts = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        sol = model(
            square_tiny_hypergraph,
            t0=0.0,
            t1=1.0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        assert sol.ys.shape == (5, 4, 8)

    def test_grad_through_ode(self, square_tiny_hypergraph, prng_key):
        """Gradients should flow through the ODE solve."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(square_tiny_hypergraph, t0=0.0, t1=0.5)
            return jnp.sum(sol.ys[-1])

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        # Check grads on the conv weight are non-zero
        assert not jnp.allclose(grads.drift.conv.linear.weight, 0.0)

    def test_jit(self, square_tiny_hypergraph, prng_key):
        """Neural ODE should work under JIT."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)

        @eqx.filter_jit
        def integrate(m, hg):
            sol = m(hg, t0=0.0, t1=0.5)
            return sol.ys[-1]

        out_eager = model(square_tiny_hypergraph, t0=0.0, t1=0.5).ys[-1]
        out_jit = integrate(model, square_tiny_hypergraph)
        assert jnp.allclose(out_eager, out_jit, atol=1e-5)

    def test_with_thnn(self, square_tiny_hypergraph, prng_key):
        """Neural ODE works with THNN conv layer."""
        conv = hgx.THNNConv(in_dim=8, out_dim=8, rank=16, key=prng_key)
        model = HypergraphNeuralODE(conv)
        sol = model(square_tiny_hypergraph, t0=0.0, t1=0.5)
        assert sol.ys.shape == (1, 4, 8)
        assert jnp.all(jnp.isfinite(sol.ys))

    def test_with_masked_hypergraph(self, prng_key):
        """Neural ODE works with dynamic topology (masked nodes)."""
        hg = hgx.from_edge_list(
            [(0, 1, 2), (1, 2, 3)],
            node_features=jnp.ones((4, 8)),
        )
        hg_pre = hgx.preallocate(hg, 6, 3)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)
        sol = model(hg_pre, t0=0.0, t1=0.5)
        # 6 total nodes (4 active + 2 preallocated)
        assert sol.ys.shape == (1, 6, 8)


# ---------------------------------------------------------------------------
# Neural SDE tests
# ---------------------------------------------------------------------------


class TestHypergraphNeuralSDE:
    """Test Neural SDE on hypergraphs."""

    def test_output_shape(self, square_tiny_hypergraph, prng_key):
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k2)
        sol = model(square_tiny_hypergraph, t0=0.0, t1=0.5, key=k2)
        # SDE state is flat: (1, n*d) = (1, 32)
        assert sol.ys.shape == (1, 32)
        # Reshape works
        assert sol.ys[-1].reshape(4, 8).shape == (4, 8)

    def test_stochasticity(self, square_tiny_hypergraph, prng_key):
        """Different keys should give different trajectories."""
        k1, k2, k3 = jax.random.split(prng_key, 3)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k1)
        sol_a = model(square_tiny_hypergraph, t0=0.0, t1=0.5, key=k2)
        sol_b = model(square_tiny_hypergraph, t0=0.0, t1=0.5, key=k3)
        assert not jnp.allclose(sol_a.ys[-1], sol_b.ys[-1])

    def test_grad_through_sde(self, square_tiny_hypergraph, prng_key):
        """Gradients should flow through the SDE solve."""
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k1)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(square_tiny_hypergraph, t0=0.0, t1=0.1, key=k2)
            return jnp.sum(sol.ys[-1])

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        assert not jnp.allclose(grads.drift.conv.linear.weight, 0.0)

    def test_diffusion_learnable(self, square_tiny_hypergraph, prng_key):
        """Diffusion parameters should receive gradients."""
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k1)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(square_tiny_hypergraph, t0=0.0, t1=0.1, key=k2)
            return jnp.sum(sol.ys[-1])

        _, grads = loss_fn(model)
        assert not jnp.allclose(grads.diffusion.log_sigma, 0.0)

    def test_jit(self, square_tiny_hypergraph, prng_key):
        """Neural SDE should work under JIT."""
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k1)

        @eqx.filter_jit
        def integrate(m, hg, key):
            sol = m(hg, t0=0.0, t1=0.1, key=key)
            return sol.ys[-1]

        out = integrate(model, square_tiny_hypergraph, k2)
        assert out.shape == (32,)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# evolve() convenience function
# ---------------------------------------------------------------------------


class TestEvolve:
    """Test the evolve convenience function."""

    def test_ode_evolve(self, square_tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)
        hg_out = evolve(model, square_tiny_hypergraph, t0=0.0, t1=0.5)
        assert isinstance(hg_out, hgx.Hypergraph)
        assert hg_out.node_features.shape == (4, 8)
        # Structure preserved
        assert jnp.array_equal(
            hg_out.incidence, square_tiny_hypergraph.incidence
        )

    def test_sde_evolve(self, square_tiny_hypergraph, prng_key):
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k1)
        hg_out = evolve(
            model, square_tiny_hypergraph, t0=0.0, t1=0.5, key=k2
        )
        assert isinstance(hg_out, hgx.Hypergraph)
        assert hg_out.node_features.shape == (4, 8)

    def test_sde_evolve_requires_key(self, square_tiny_hypergraph, prng_key):
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k1)
        with pytest.raises(ValueError, match="Must provide key"):
            evolve(model, square_tiny_hypergraph)


# ---------------------------------------------------------------------------
# trajectory() convenience function
# ---------------------------------------------------------------------------


class TestTrajectory:
    """Test the trajectory convenience function."""

    def test_trajectory_ode_shape(self, square_tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = HypergraphNeuralODE(conv)
        ts, features = trajectory(model, square_tiny_hypergraph, num_steps=10)
        assert ts.shape == (10,)
        assert features.shape == (10, 4, 8)

    def test_trajectory_sde_shape(self, square_tiny_hypergraph, prng_key):
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=k1)
        model = HypergraphNeuralSDE(conv, num_nodes=4, node_dim=8, key=k1)
        ts, features = trajectory(
            model, square_tiny_hypergraph, num_steps=10, key=k2
        )
        assert ts.shape == (10,)
        assert features.shape == (10, 4, 8)
