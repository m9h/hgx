"""Tests for continuous-time dynamics on hypergraphs."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._dynamics import (
    evolve,
    HypergraphNeuralCDE,
    HypergraphNeuralODE,
    HypergraphNeuralSDE,
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
# Neural CDE tests
# ---------------------------------------------------------------------------


class TestHypergraphNeuralCDE:
    """Test Neural CDE on hypergraphs."""

    def test_output_shape(self, square_tiny_hypergraph, prng_key):
        """Verify (T, n, d) trajectory shape."""
        # Conv maps 8 -> 8*3 = 24 for control_dim=3
        conv = hgx.UniGCNConv(in_dim=8, out_dim=24, key=prng_key)
        model = HypergraphNeuralCDE(conv, control_dim=3)

        ts = jnp.linspace(0.0, 1.0, 5)
        controls = jnp.ones((5, 4, 3))

        sol = model(square_tiny_hypergraph, ts=ts, controls=controls)
        assert sol.ys.shape == (5, 4, 8)

    def test_grad_through_cde(self, square_tiny_hypergraph, prng_key):
        """Gradients should flow to conv weights."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=24, key=prng_key)
        model = HypergraphNeuralCDE(conv, control_dim=3)

        ts = jnp.linspace(0.0, 1.0, 5)
        # Time-varying controls so dX/dt != 0
        controls = ts[:, None, None] * jnp.ones((1, 4, 3))

        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(square_tiny_hypergraph, ts=ts, controls=controls)
            return jnp.sum(sol.ys[-1])

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        assert not jnp.allclose(grads.drift.conv.linear.weight, 0.0)

    def test_jit(self, square_tiny_hypergraph, prng_key):
        """Neural CDE should work under eqx.filter_jit."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=24, key=prng_key)
        model = HypergraphNeuralCDE(conv, control_dim=3)

        ts = jnp.linspace(0.0, 1.0, 5)
        controls = jnp.ones((5, 4, 3))

        @eqx.filter_jit
        def integrate(m, hg, ts, controls):
            sol = m(hg, ts=ts, controls=controls)
            return sol.ys[-1]

        out_eager = model(
            square_tiny_hypergraph, ts=ts, controls=controls
        ).ys[-1]
        out_jit = integrate(model, square_tiny_hypergraph, ts, controls)
        assert jnp.allclose(out_eager, out_jit, atol=1e-5)

    def test_control_affects_output(self, square_tiny_hypergraph, prng_key):
        """Different control signals should produce different outputs."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=24, key=prng_key)
        model = HypergraphNeuralCDE(conv, control_dim=3)

        ts = jnp.linspace(0.0, 1.0, 5)
        # Two different time-varying control patterns
        controls_a = ts[:, None, None] * jnp.ones((1, 4, 3))
        controls_b = (1.0 - ts[:, None, None]) * jnp.ones((1, 4, 3))

        sol_a = model(square_tiny_hypergraph, ts=ts, controls=controls_a)
        sol_b = model(square_tiny_hypergraph, ts=ts, controls=controls_b)
        assert not jnp.allclose(sol_a.ys[-1], sol_b.ys[-1])


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


# ---------------------------------------------------------------------------
# Convergence / mathematical validation
# ---------------------------------------------------------------------------


def _identity(x):
    """Identity activation (module-level for stable JIT caching)."""
    return x


class TestDynamicsConvergence:
    """Mathematical validation of ODE/SDE convergence properties."""

    def test_ode_convergence_order(self, prng_key):
        """Tighter ODE tolerance gives closer match to analytical solution.

        Setup: diagonal hypergraph (H=I) with conv W=-I, no bias,
        no normalization, identity activation => dx/dt = -x.
        Analytical: x(t) = x0 * exp(-t).
        """
        import diffrax

        n, d = 3, 2
        H = jnp.eye(n)
        x0 = jnp.array([[1.0, 2.0], [0.5, 1.5], [3.0, 0.5]])
        hg = hgx.from_incidence(H, node_features=x0)

        # Conv with W = -I, no bias => conv(hg) = -x
        conv = hgx.UniGCNConv(
            in_dim=d, out_dim=d, use_bias=False, normalize=False, key=prng_key
        )
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, -jnp.eye(d))

        t1 = 1.0
        x_exact = x0 * jnp.exp(-t1)

        # Coarse tolerance
        model_coarse = HypergraphNeuralODE(
            conv,
            activation=_identity,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        )
        err_coarse = jnp.max(jnp.abs(
            model_coarse(hg, t0=0.0, t1=t1).ys[-1] - x_exact
        ))

        # Tight tolerance
        model_tight = HypergraphNeuralODE(
            conv,
            activation=_identity,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        )
        err_tight = jnp.max(jnp.abs(
            model_tight(hg, t0=0.0, t1=t1).ys[-1] - x_exact
        ))

        assert err_tight < err_coarse
        assert err_tight < 1e-4

    def test_sde_mean_matches_drift(self, prng_key):
        """Mean of many SDE trajectories approximates ODE solution (LLN).

        Uses small sigma so noise is a minor perturbation on the drift.
        """
        n, d = 3, 2
        H = jnp.eye(n)
        x0 = jnp.ones((n, d))
        hg = hgx.from_incidence(H, node_features=x0)

        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(
            in_dim=d, out_dim=d, use_bias=False, normalize=False, key=k1
        )
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, -0.5 * jnp.eye(d))

        t1 = 0.3

        # ODE reference solution (high-accuracy Tsit5)
        x_ode = HypergraphNeuralODE(
            conv, activation=_identity,
        )(hg, t0=0.0, t1=t1).ys[-1]

        # SDE: 32 samples with small sigma
        model_sde = HypergraphNeuralSDE(
            conv, num_nodes=n, node_dim=d, activation=_identity,
            sigma_init=0.01, dt=0.01, key=k2,
        )

        @eqx.filter_jit
        def solve_one(model, hg, key):
            return model(hg, t0=0.0, t1=0.3, key=key).ys[-1].reshape(n, d)

        num_samples = 32
        keys = jax.random.split(k2, num_samples)
        endpoints = jnp.stack([solve_one(model_sde, hg, ki) for ki in keys])
        mean_sde = jnp.mean(endpoints, axis=0)

        assert jnp.allclose(mean_sde, x_ode, atol=0.05)

    def test_sde_variance_scales_with_sigma(self, prng_key):
        """Variance of SDE endpoints scales with sigma^2.

        sigma_large / sigma_small = 10 => Var_large / Var_small ~ 100.
        """
        n, d = 3, 2
        H = jnp.eye(n)
        x0 = jnp.ones((n, d))
        hg = hgx.from_incidence(H, node_features=x0)

        k1, k2, k3 = jax.random.split(prng_key, 3)
        conv = hgx.UniGCNConv(
            in_dim=d, out_dim=d, use_bias=False, normalize=False, key=k1
        )
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, -0.5 * jnp.eye(d))

        num_samples = 32

        model_small = HypergraphNeuralSDE(
            conv, num_nodes=n, node_dim=d, activation=_identity,
            sigma_init=0.01, dt=0.01, key=k2,
        )
        model_large = HypergraphNeuralSDE(
            conv, num_nodes=n, node_dim=d, activation=_identity,
            sigma_init=0.1, dt=0.01, key=k3,
        )

        @eqx.filter_jit
        def solve(model, hg, key):
            return model(hg, t0=0.0, t1=0.5, key=key).ys[-1]

        keys_s = jax.random.split(k2, num_samples)
        keys_l = jax.random.split(k3, num_samples)

        ep_small = jnp.stack([solve(model_small, hg, ki) for ki in keys_s])
        ep_large = jnp.stack([solve(model_large, hg, ki) for ki in keys_l])

        var_small = jnp.mean(jnp.var(ep_small, axis=0))
        var_large = jnp.mean(jnp.var(ep_large, axis=0))

        # Theory: ratio ~ (0.1/0.01)^2 = 100; conservative threshold
        ratio = var_large / (var_small + 1e-12)
        assert ratio > 20

    def test_energy_conservation_short_time(self, prng_key):
        """Short ODE integration with tanh-bounded drift preserves norm."""
        n, d = 4, 8
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
        k1, k2 = jax.random.split(prng_key)
        x0 = jax.random.normal(k1, (n, d))
        hg = hgx.from_incidence(H, node_features=x0)

        conv = hgx.UniGCNConv(in_dim=d, out_dim=d, key=k2)
        model = HypergraphNeuralODE(conv)  # default tanh activation

        sol = model(hg, t0=0.0, t1=0.01)
        x_final = sol.ys[-1]

        norm_init = jnp.linalg.norm(x0)
        norm_final = jnp.linalg.norm(x_final)

        relative_change = jnp.abs(norm_final - norm_init) / norm_init
        assert relative_change < 0.1
