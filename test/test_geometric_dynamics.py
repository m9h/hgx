"""Tests for geometric dynamics on Riemannian manifolds."""

import pytest


pytest.importorskip("diffrax")

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
from hgx._dynamics import HypergraphNeuralODE
from hgx._geometric_dynamics import (
    EuclideanManifold,
    PoincareBall,
    riemannian_trajectory,
    RiemannianHypergraphODE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def poincare_hypergraph(prng_key):
    """Hypergraph with features inside the Poincaré ball (norm < 1)."""
    H = jnp.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    features = 0.3 * jax.random.normal(prng_key, (4, 8))
    return hgx.from_incidence(H, node_features=features)


@pytest.fixture
def euclidean_hypergraph():
    """Hypergraph for Euclidean tests."""
    H = jnp.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    features = jnp.ones((4, 8))
    return hgx.from_incidence(H, node_features=features)


# ---------------------------------------------------------------------------
# Poincaré ball unit tests
# ---------------------------------------------------------------------------


class TestPoincareBall:
    """Test Poincaré ball manifold operations."""

    def test_project_inside(self):
        """Points inside ball are unchanged."""
        ball = PoincareBall(c=1.0)
        x = jnp.array([[0.3, 0.4], [-0.1, 0.2]])
        assert jnp.allclose(ball.project(x), x, atol=1e-6)

    def test_project_outside(self):
        """Points outside ball are projected onto boundary."""
        ball = PoincareBall(c=1.0)
        x = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        proj = ball.project(x)
        norms = jnp.linalg.norm(proj, axis=-1)
        assert jnp.all(norms < 1.0)

    def test_expmap_at_origin(self):
        """Exp map at origin: exp_0(v) = tanh(||v||) * v/||v||."""
        ball = PoincareBall(c=1.0)
        origin = jnp.zeros((1, 3))
        v = jnp.array([[0.5, 0.0, 0.0]])
        result = ball.expmap(origin, v)
        expected_norm = jnp.tanh(0.5)
        assert jnp.allclose(
            jnp.linalg.norm(result), expected_norm, atol=1e-4
        )

    def test_expmap_stays_in_ball(self):
        """Exp map always produces points inside the ball."""
        ball = PoincareBall(c=1.0)
        x = jnp.array([[0.5, 0.3], [-0.2, 0.4]])
        v = jnp.array([[1.0, -1.0], [2.0, 0.5]])
        result = ball.expmap(x, v)
        norms = jnp.linalg.norm(result, axis=-1)
        assert jnp.all(norms < 1.0)

    def test_logmap_inverts_expmap(self):
        """Log map is the inverse of exp map."""
        ball = PoincareBall(c=1.0)
        x = jnp.array([[0.1, 0.2], [-0.3, 0.1]])
        v = jnp.array([[0.1, -0.05], [0.05, 0.1]])
        y = ball.expmap(x, v)
        v_recovered = ball.logmap(x, y)
        assert jnp.allclose(v, v_recovered, atol=1e-3)

    def test_custom_curvature(self):
        """Projection respects curvature parameter c."""
        ball = PoincareBall(c=4.0)  # ball radius = 1/sqrt(4) = 0.5
        x = jnp.array([[0.6, 0.0]])
        proj = ball.project(x)
        assert jnp.linalg.norm(proj) < 0.5


# ---------------------------------------------------------------------------
# RiemannianHypergraphODE tests
# ---------------------------------------------------------------------------


class TestRiemannianHypergraphODE:
    """Test Neural ODE on Riemannian manifolds."""

    def test_poincare_stays_on_manifold(self, poincare_hypergraph, prng_key):
        """Trajectory stays inside Poincaré ball throughout integration."""
        import diffrax

        ball = PoincareBall(c=1.0)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=ball)

        ts = jnp.linspace(0.0, 1.0, 20)
        sol = model(
            poincare_hypergraph,
            t0=0.0,
            t1=1.0,
            saveat=diffrax.SaveAt(ts=ts),
        )

        # Every saved state must be inside the unit ball
        for i in range(sol.ys.shape[0]):
            norms = jnp.linalg.norm(sol.ys[i], axis=-1)
            assert jnp.all(norms < 1.0), (
                f"Frame {i}: max norm = {float(jnp.max(norms)):.6f}"
            )

    def test_poincare_output_shape(self, poincare_hypergraph, prng_key):
        """Output shape is correct for Poincaré ODE."""
        ball = PoincareBall(c=1.0)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=ball)

        sol = model(poincare_hypergraph, t0=0.0, t1=1.0)
        assert sol.ys.shape == (1, 4, 8)

    def test_euclidean_matches_flat_ode(self, euclidean_hypergraph, prng_key):
        """Euclidean Riemannian ODE matches standard HypergraphNeuralODE."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)

        model_flat = HypergraphNeuralODE(conv)
        model_riem = RiemannianHypergraphODE(
            conv, manifold=EuclideanManifold()
        )

        sol_flat = model_flat(euclidean_hypergraph, t0=0.0, t1=0.5)
        sol_riem = model_riem(euclidean_hypergraph, t0=0.0, t1=0.5)

        assert jnp.allclose(sol_flat.ys, sol_riem.ys, atol=1e-5)

    def test_jit_poincare(self, poincare_hypergraph, prng_key):
        """Poincaré ODE works under JIT."""
        ball = PoincareBall(c=1.0)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=ball)

        @eqx.filter_jit
        def integrate(m, hg):
            sol = m(hg, t0=0.0, t1=0.5)
            return sol.ys[-1]

        out_eager = model(poincare_hypergraph, t0=0.0, t1=0.5).ys[-1]
        out_jit = integrate(model, poincare_hypergraph)
        assert jnp.allclose(out_eager, out_jit, atol=1e-5)

    def test_jit_euclidean(self, euclidean_hypergraph, prng_key):
        """Euclidean Riemannian ODE works under JIT."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=EuclideanManifold())

        @eqx.filter_jit
        def integrate(m, hg):
            sol = m(hg, t0=0.0, t1=0.5)
            return sol.ys[-1]

        out_eager = model(euclidean_hypergraph, t0=0.0, t1=0.5).ys[-1]
        out_jit = integrate(model, euclidean_hypergraph)
        assert jnp.allclose(out_eager, out_jit, atol=1e-5)

    def test_grad_through_poincare_ode(self, poincare_hypergraph, prng_key):
        """Gradients flow through Poincaré ODE solve."""
        ball = PoincareBall(c=1.0)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=ball)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(poincare_hypergraph, t0=0.0, t1=0.5)
            return jnp.sum(sol.ys[-1])

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        assert not jnp.allclose(grads.drift.conv.linear.weight, 0.0)

    def test_features_change(self, poincare_hypergraph, prng_key):
        """Integration should change features (non-trivial dynamics)."""
        ball = PoincareBall(c=1.0)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=ball)

        sol = model(poincare_hypergraph, t0=0.0, t1=1.0)
        y_final = sol.ys[-1]
        y_init = ball.project(poincare_hypergraph.node_features)
        assert not jnp.allclose(y_final, y_init)

    def test_default_euclidean(self, euclidean_hypergraph, prng_key):
        """Default manifold is Euclidean when not specified."""
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv)
        assert isinstance(model.manifold, EuclideanManifold)

        sol = model(euclidean_hypergraph, t0=0.0, t1=0.5)
        assert sol.ys.shape == (1, 4, 8)


# ---------------------------------------------------------------------------
# riemannian_trajectory tests
# ---------------------------------------------------------------------------


class TestRiemannianTrajectory:
    """Test riemannian_trajectory convenience function."""

    def test_shape(self, poincare_hypergraph, prng_key):
        """Trajectory has correct shape."""
        ball = PoincareBall(c=1.0)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=ball)

        ts, features = riemannian_trajectory(
            model, poincare_hypergraph, num_steps=10
        )
        assert ts.shape == (10,)
        assert features.shape == (10, 4, 8)

    def test_on_manifold(self, poincare_hypergraph, prng_key):
        """All trajectory points are on the manifold."""
        ball = PoincareBall(c=1.0)
        conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=prng_key)
        model = RiemannianHypergraphODE(conv, manifold=ball)

        _, features = riemannian_trajectory(
            model, poincare_hypergraph, num_steps=20
        )
        norms = jnp.linalg.norm(features, axis=-1)  # (20, 4)
        assert jnp.all(norms < 1.0)
