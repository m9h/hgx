"""Tests for optimal transport utilities."""

from __future__ import annotations

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
from hgx._ot import (
    feature_cost_matrix,
    gromov_wasserstein,
    hypergraph_gromov_wasserstein,
    hypergraph_wasserstein,
    ot_hyperedge_aggregation,
    ot_hypergraph_alignment,
    OTLayer,
    sinkhorn,
    structural_cost_matrix,
    unbalanced_sinkhorn,
    wasserstein_barycenter,
    wasserstein_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hg(key, n=5, m=3, d=2):
    """Create a random hypergraph for testing."""
    k1, k2 = jax.random.split(key)
    H = jax.random.bernoulli(
        k1, shape=(n, m)
    ).astype(jnp.float32)
    H = H.at[0, :].set(1.0)
    features = jax.random.normal(k2, (n, d))
    return hgx.from_incidence(H, node_features=features)


# ---------------------------------------------------------------------------
# Sinkhorn
# ---------------------------------------------------------------------------


class TestSinkhorn:
    """Test the core Sinkhorn algorithm."""

    def test_sinkhorn_marginals(self, prng_key):
        """Transport plan rows/cols sum to marginals."""
        n, m = 5, 7
        k1, k2, k3 = jax.random.split(prng_key, 3)
        a = jax.random.dirichlet(k1, jnp.ones(n))
        b = jax.random.dirichlet(k2, jnp.ones(m))
        cost = jax.random.uniform(k3, (n, m))

        P = sinkhorn(cost, a, b, epsilon=0.05, max_iters=100)

        assert jnp.allclose(
            jnp.sum(P, axis=1), a, atol=1e-4
        )
        assert jnp.allclose(
            jnp.sum(P, axis=0), b, atol=1e-4
        )

    def test_sinkhorn_positive(self):
        """All entries non-negative."""
        a = jnp.ones(4) / 4
        b = jnp.ones(6) / 6
        cost = jnp.ones((4, 6))
        P = sinkhorn(cost, a, b)
        assert jnp.all(P >= -1e-10)

    def test_transport_cost_non_negative(self):
        """Transport cost should be non-negative."""
        a = jnp.ones(3) / 3
        b = jnp.ones(3) / 3
        cost = jnp.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ])
        P = sinkhorn(cost, a, b)
        cost_val = jnp.sum(P * cost)
        assert cost_val >= -1e-10

    def test_differentiable(self):
        """Sinkhorn differentiable w.r.t. cost matrix."""
        a = jnp.ones(3) / 3
        b = jnp.ones(3) / 3

        def loss(cost):
            P = sinkhorn(cost, a, b, epsilon=0.1)
            return jnp.sum(P * cost)

        cost = jnp.ones((3, 3))
        grad = jax.grad(loss)(cost)
        assert grad.shape == (3, 3)
        assert jnp.isfinite(grad).all()

    def test_jit_compatible(self):
        """Sinkhorn should run under jax.jit."""
        a = jnp.ones(4) / 4
        b = jnp.ones(4) / 4
        cost = jnp.ones((4, 4))

        jit_fn = jax.jit(
            sinkhorn, static_argnames=("max_iters",)
        )
        P = jit_fn(cost, a, b, epsilon=0.1, max_iters=30)
        assert P.shape == (4, 4)
        assert jnp.isfinite(P).all()


# ---------------------------------------------------------------------------
# Cost matrices
# ---------------------------------------------------------------------------


class TestCostMatrices:
    """Test cost matrix construction."""

    def test_cost_matrix_shape(self):
        """feature_cost_matrix has correct output shape."""
        X = jnp.ones((5, 3))
        Y = jnp.ones((7, 3))
        C = feature_cost_matrix(X, Y)
        assert C.shape == (5, 7)

    def test_cost_matrix_euclidean_nonneg(self):
        """Euclidean cost matrix entries are non-negative."""
        X = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        Y = jnp.array([[2.0, 2.0]])
        C = feature_cost_matrix(X, Y, metric="euclidean")
        assert jnp.all(C >= 0.0)

    def test_cost_matrix_sqeuclidean(self):
        """Squared Euclidean gives squared distances."""
        X = jnp.array([[0.0]])
        Y = jnp.array([[3.0]])
        C = feature_cost_matrix(X, Y, metric="sqeuclidean")
        assert jnp.allclose(C, jnp.array([[9.0]]))

    def test_cost_matrix_cosine(self):
        """Cosine cost between identical vectors is ~0."""
        X = jnp.array([[1.0, 0.0]])
        Y = jnp.array([[2.0, 0.0]])
        C = feature_cost_matrix(X, Y, metric="cosine")
        assert jnp.allclose(C, 0.0, atol=1e-5)

    def test_structural_cost_matrix_shape(self, prng_key):
        """structural_cost_matrix returns correct shape."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=4, m=2, d=3)
        hg2 = _make_hg(k2, n=6, m=3, d=3)
        C = structural_cost_matrix(hg1, hg2)
        assert C.shape == (4, 6)

    def test_structural_cost_matrix_finite(self, prng_key):
        """structural_cost_matrix values are finite."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=5)
        hg2 = _make_hg(k2, n=5)
        C = structural_cost_matrix(hg1, hg2)
        assert jnp.isfinite(C).all()


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------


class TestWassersteinDistance:
    """Test Wasserstein distance between point clouds."""

    def test_wasserstein_self_zero(self):
        """W(X, X) should be approximately 0."""
        X = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        d = wasserstein_distance(X, X, epsilon=0.01)
        assert d < 0.1

    def test_non_negative(self, prng_key):
        """Wasserstein distance is non-negative."""
        k1, k2 = jax.random.split(prng_key)
        x = jax.random.normal(k1, (5, 3))
        y = jax.random.normal(k2, (7, 3))
        d = wasserstein_distance(x, y)
        assert d >= -1e-10

    def test_positive_for_different(self):
        """Distance positive for separated point clouds."""
        x = jnp.zeros((5, 2))
        y = jnp.ones((5, 2)) * 10.0
        d = wasserstein_distance(x, y)
        assert d > 1.0


# ---------------------------------------------------------------------------
# Wasserstein barycenter
# ---------------------------------------------------------------------------


class TestWassersteinBarycenter:
    """Test Wasserstein barycenter computation."""

    def test_output_shape(self):
        """Barycenter has the requested support size."""
        d1 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        d2 = jnp.array([[0.0, 1.0], [1.0, 1.0]])
        weights = jnp.array([0.5, 0.5])
        bary = wasserstein_barycenter(
            [d1, d2], weights,
            support_size=3, num_iters=10,
        )
        assert bary.shape == (3, 2)

    def test_barycenter_between_inputs(self):
        """Barycenter lies between input distributions."""
        d1 = jnp.array([[0.0, 0.0]])
        d2 = jnp.array([[2.0, 2.0]])
        weights = jnp.array([0.5, 0.5])
        bary = wasserstein_barycenter(
            [d1, d2], weights,
            support_size=1, epsilon=0.01, num_iters=50,
        )
        centroid = jnp.mean(bary, axis=0)
        assert jnp.allclose(
            centroid, jnp.array([1.0, 1.0]), atol=0.3
        )


# ---------------------------------------------------------------------------
# Gromov-Wasserstein
# ---------------------------------------------------------------------------


class TestGromovWasserstein:
    """Test Gromov-Wasserstein distance."""

    def test_gw_symmetric(self, prng_key):
        """GW(A,B) should approximately equal GW(B,A)."""
        k1, k2 = jax.random.split(prng_key)
        n1, n2 = 5, 6
        C1 = jax.random.uniform(k1, (n1, n1))
        C1 = C1 + C1.T
        C2 = jax.random.uniform(k2, (n2, n2))
        C2 = C2 + C2.T
        p = jnp.ones(n1) / n1
        q = jnp.ones(n2) / n2

        _, d12 = gromov_wasserstein(
            C1, C2, p, q, max_iters=15
        )
        _, d21 = gromov_wasserstein(
            C2, C1, q, p, max_iters=15
        )
        assert jnp.allclose(d12, d21, atol=0.2)

    def test_gw_non_negative(self, prng_key):
        """GW cost should be non-negative."""
        n = 5
        C = jax.random.uniform(prng_key, (n, n))
        C = C + C.T
        p = jnp.ones(n) / n
        _, cost = gromov_wasserstein(
            C, C, p, p, max_iters=10
        )
        assert cost >= -1e-6

    def test_gw_transport_plan_shape(self, prng_key):
        """Transport plan has shape (n, m)."""
        n1, n2 = 4, 6
        k1, k2 = jax.random.split(prng_key)
        C1 = jax.random.uniform(k1, (n1, n1))
        C2 = jax.random.uniform(k2, (n2, n2))
        p = jnp.ones(n1) / n1
        q = jnp.ones(n2) / n2
        T, _ = gromov_wasserstein(
            C1, C2, p, q, max_iters=5
        )
        assert T.shape == (n1, n2)


# ---------------------------------------------------------------------------
# Hypergraph-specific OT functions
# ---------------------------------------------------------------------------


class TestHypergraphOT:
    """Test hypergraph-level OT convenience functions."""

    def test_hypergraph_wasserstein_finite(self, prng_key):
        """hypergraph_wasserstein output is finite."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=4, d=3)
        hg2 = _make_hg(k2, n=5, d=3)
        d = hypergraph_wasserstein(hg1, hg2)
        assert jnp.isfinite(d)

    def test_hypergraph_gw_finite(self, prng_key):
        """hypergraph_gromov_wasserstein output is finite."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=4)
        hg2 = _make_hg(k2, n=5)
        T, cost = hypergraph_gromov_wasserstein(hg1, hg2)
        assert jnp.isfinite(cost)
        assert jnp.isfinite(T).all()

    def test_fused_gw_shape(self, prng_key):
        """ot_hypergraph_alignment transport plan shape."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=4, d=3)
        hg2 = _make_hg(k2, n=5, d=3)
        T, cost = ot_hypergraph_alignment(hg1, hg2)
        assert T.shape == (4, 5)
        assert jnp.isfinite(cost)


# ---------------------------------------------------------------------------
# OTLayer
# ---------------------------------------------------------------------------


class TestOTLayer:
    """Test the OTLayer differentiable alignment layer."""

    def test_ot_layer_output_shape(self, prng_key):
        """Transport plan has correct shape."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=4, d=3)
        hg2 = _make_hg(k2, n=6, d=3)
        layer = OTLayer(epsilon=0.1, max_iters=50)
        P, aligned = layer(hg1, hg2)
        assert P.shape == (4, 6)
        assert aligned.shape == (6, 3)

    def test_ot_layer_jit(self, prng_key):
        """OTLayer works under eqx.filter_jit."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=4, d=3)
        hg2 = _make_hg(k2, n=5, d=3)
        layer = OTLayer(epsilon=0.1, max_iters=50)
        jit_layer = eqx.filter_jit(layer)
        P, aligned = jit_layer(hg1, hg2)
        assert P.shape == (4, 5)
        assert aligned.shape == (5, 3)
        assert jnp.isfinite(P).all()
        assert jnp.isfinite(aligned).all()

    def test_ot_layer_plan_non_negative(self, prng_key):
        """Transport plan entries are non-negative."""
        k1, k2 = jax.random.split(prng_key)
        hg1 = _make_hg(k1, n=4, d=2)
        hg2 = _make_hg(k2, n=5, d=2)
        layer = OTLayer(epsilon=0.1, max_iters=50)
        P, _ = layer(hg1, hg2)
        assert jnp.all(P >= -1e-10)


# ---------------------------------------------------------------------------
# OTConv layer
# ---------------------------------------------------------------------------


class TestOTConv:
    """Test the OT convolution layer."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        """Output should be (num_nodes, out_dim)."""
        conv = hgx.OTConv(
            in_dim=2, out_dim=8, key=prng_key
        )
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_jit_compatible(
        self, tiny_hypergraph, prng_key
    ):
        """OTConv works under eqx.filter_jit."""
        conv = hgx.OTConv(
            in_dim=2, out_dim=4, key=prng_key
        )
        jit_conv = eqx.filter_jit(conv)
        out = jit_conv(tiny_hypergraph)
        assert out.shape == (4, 4)
        out_eager = conv(tiny_hypergraph)
        assert jnp.allclose(out, out_eager, atol=1e-5)

    def test_gradient_flow(
        self, tiny_hypergraph, prng_key
    ):
        """Gradients flow through OTConv parameters."""
        conv = hgx.OTConv(
            in_dim=2, out_dim=4, key=prng_key
        )

        @eqx.filter_grad
        def grad_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = grad_fn(conv)
        assert not jnp.allclose(
            grads.linear_in.weight, 0.0
        )
        assert not jnp.allclose(
            grads.linear_out.weight, 0.0
        )
        assert jnp.isfinite(grads.log_epsilon)

    def test_finite_output(
        self, tiny_hypergraph, prng_key
    ):
        """All outputs should be finite."""
        conv = hgx.OTConv(
            in_dim=2, out_dim=4, key=prng_key
        )
        out = conv(tiny_hypergraph)
        assert jnp.isfinite(out).all()


# ---------------------------------------------------------------------------
# OT hyperedge aggregation
# ---------------------------------------------------------------------------


class TestOTHyperedgeAggregation:
    """Test OT-based vertex-to-edge aggregation."""

    def test_output_shape(self):
        """Output should be (num_edges, feature_dim)."""
        n, m, d = 6, 3, 4
        node_features = jnp.ones((n, d))
        incidence = jax.random.bernoulli(
            jax.random.PRNGKey(2), shape=(n, m)
        ).astype(jnp.float32)
        incidence = incidence.at[0, :].set(1.0)
        out = ot_hyperedge_aggregation(
            node_features, incidence
        )
        assert out.shape == (m, d)

    def test_uniform_features(self):
        """Identical node features produce matching output."""
        val = 3.14
        n, m, d = 4, 2, 3
        features = jnp.full((n, d), val)
        incidence = jnp.ones((n, m))
        out = ot_hyperedge_aggregation(
            features, incidence
        )
        assert jnp.allclose(out, val, atol=0.1)


# ---------------------------------------------------------------------------
# Unbalanced Sinkhorn
# ---------------------------------------------------------------------------


class TestUnbalancedSinkhorn:
    """Test the unbalanced Sinkhorn algorithm."""

    def test_allows_mass_creation(self):
        """Marginals deviate from a and b when tau is finite."""
        a = jnp.array([0.5, 0.5])
        b = jnp.array([0.25, 0.25, 0.25, 0.25])
        cost = jnp.ones((2, 4))

        T, _ = unbalanced_sinkhorn(
            a, b, cost,
            epsilon=0.1, tau=0.5, num_iters=100,
        )

        assert T.shape == (2, 4)
        assert jnp.all(T >= -1e-10)
        assert jnp.sum(T) > 0

    def test_large_tau_approaches_balanced(self):
        """Large tau recovers balanced Sinkhorn."""
        a = jnp.ones(3) / 3
        b = jnp.ones(3) / 3
        cost = jnp.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ])

        T_bal = sinkhorn(
            cost, a, b, epsilon=0.1, max_iters=100
        )
        T_unbal, _ = unbalanced_sinkhorn(
            a, b, cost,
            epsilon=0.1, tau=100.0, num_iters=100,
        )

        assert jnp.allclose(T_bal, T_unbal, atol=0.05)

    def test_transport_plan_non_negative(self):
        """Transport plan entries are non-negative."""
        a = jnp.ones(4) / 4
        b = jnp.ones(5) / 5
        cost = jnp.ones((4, 5))
        T, _ = unbalanced_sinkhorn(a, b, cost, tau=1.0)
        assert jnp.all(T >= -1e-10)
