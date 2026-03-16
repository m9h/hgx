"""Tests for information-geometric neural dynamics on hypergraphs."""

from __future__ import annotations

import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._info_geometry import (
    fisher_rao_distance,
    fisher_rao_metric,
    FisherRaoDrift,
    free_energy,
    FreeEnergyDrift,
    info_belief_update,
    InfoGeometricDynamics,
    InfoGeometricODE,
    js_divergence,
    kl_divergence,
    natural_gradient,
    natural_gradient_descent,
    symmetrized_kl,
    wasserstein_on_simplex,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_hypergraph():
    """Small hypergraph with 4 nodes, 2 edges, feature dim 8."""
    H = jnp.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    features = jnp.ones((4, 8))
    return hgx.from_incidence(H, node_features=features)


@pytest.fixture
def prob_vectors():
    """Two sets of probability vectors on a 4-simplex (5 nodes, K=4)."""
    p = jax.nn.softmax(jnp.array([
        [1.0, 2.0, 0.5, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        [3.0, 1.0, 0.1, 0.5],
    ]), axis=-1)
    q = jax.nn.softmax(jnp.array([
        [0.5, 0.5, 1.0, 2.0],
        [0.4, 0.3, 0.2, 0.1],
        [1.0, 1.0, 1.0, 1.0],
    ]), axis=-1)
    return p, q


# ---------------------------------------------------------------------------
# Fisher-Rao metric tests
# ---------------------------------------------------------------------------


class TestFisherRaoMetric:
    """Test Fisher-Rao metric tensor computation."""

    def test_fisher_metric_shape(self, prob_vectors):
        """Correct output shape (n, K, K)."""
        p, _ = prob_vectors
        metric = fisher_rao_metric(p)
        assert metric.shape == (3, 4, 4)

    def test_fisher_metric_positive_definite(self, prob_vectors):
        """Eigenvalues of the Fisher metric should all be positive."""
        p, _ = prob_vectors
        metric = fisher_rao_metric(p)
        for i in range(p.shape[0]):
            eigenvalues = jnp.linalg.eigvalsh(metric[i])
            assert jnp.all(eigenvalues > 0), (
                f"Row {i}: eigenvalues {eigenvalues} not all positive"
            )

    def test_diagonal(self, prob_vectors):
        """Metric for categorical should be purely diagonal."""
        p, _ = prob_vectors
        metric = fisher_rao_metric(p)
        for i in range(p.shape[0]):
            off_diag = metric[i] - jnp.diag(jnp.diag(metric[i]))
            assert jnp.allclose(off_diag, 0.0)

    def test_values(self):
        """Check that g_kk = 1/p_k for a known distribution."""
        p = jnp.array([[0.25, 0.25, 0.25, 0.25]])
        metric = fisher_rao_metric(p)
        expected_diag = jnp.array([4.0, 4.0, 4.0, 4.0])
        assert jnp.allclose(jnp.diag(metric[0]), expected_diag)

    def test_handles_zeros(self):
        """Zeros in probability vectors should be regularized."""
        p = jnp.array([[0.5, 0.5, 0.0, 0.0]])
        metric = fisher_rao_metric(p)
        assert jnp.all(jnp.isfinite(metric))
        assert jnp.all(jnp.diag(metric[0]) > 0)


# ---------------------------------------------------------------------------
# Natural gradient tests
# ---------------------------------------------------------------------------


class TestNaturalGradient:
    """Test natural gradient computation."""

    def test_natural_gradient_shape(self, prob_vectors):
        """Natural gradient should have the same shape as input."""
        p, _ = prob_vectors
        grad = jnp.ones_like(p)
        nat_grad = natural_gradient(grad, p)
        assert nat_grad.shape == p.shape

    def test_differs_from_euclidean(self, prob_vectors):
        """Natural gradient should differ from Euclidean gradient
        when probabilities are non-uniform."""
        p, _ = prob_vectors
        euclidean = jnp.ones_like(p) * 2.0
        nat_grad = natural_gradient(euclidean, p)
        assert not jnp.allclose(nat_grad, euclidean)

    def test_uniform_scaling(self):
        """For uniform dist, natural gradient = (1/K) * Euclidean."""
        K = 4
        p = jnp.ones((2, K)) / K
        euclidean = jnp.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ])
        nat_grad = natural_gradient(euclidean, p)
        expected = euclidean / K
        assert jnp.allclose(nat_grad, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# KL divergence tests
# ---------------------------------------------------------------------------


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_nonneg(self, prob_vectors):
        """KL(p, q) >= 0 (Gibbs' inequality)."""
        p, q = prob_vectors
        kl = kl_divergence(p, q)
        assert jnp.all(kl >= -1e-7), (
            f"KL should be non-negative, got {kl}"
        )

    def test_kl_self_zero(self, prob_vectors):
        """KL(p, p) should be approximately 0."""
        p, _ = prob_vectors
        kl = kl_divergence(p, p)
        assert jnp.allclose(kl, 0.0, atol=1e-6)

    def test_shape(self, prob_vectors):
        p, q = prob_vectors
        kl = kl_divergence(p, q)
        assert kl.shape == (3,)

    def test_known_value(self):
        """KL between [0.5, 0.5] and [0.25, 0.75]."""
        p = jnp.array([[0.5, 0.5]])
        q = jnp.array([[0.25, 0.75]])
        kl = kl_divergence(p, q)
        expected = (
            0.5 * jnp.log(0.5 / 0.25) + 0.5 * jnp.log(0.5 / 0.75)
        )
        assert jnp.allclose(kl[0], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Symmetrized KL tests
# ---------------------------------------------------------------------------


class TestSymmetrizedKL:
    """Test symmetrized KL divergence."""

    def test_symmetric(self, prob_vectors):
        """symmetrized_kl(p, q) == symmetrized_kl(q, p)."""
        p, q = prob_vectors
        skl_pq = symmetrized_kl(p, q)
        skl_qp = symmetrized_kl(q, p)
        assert jnp.allclose(skl_pq, skl_qp, atol=1e-6)

    def test_zero_for_identical(self, prob_vectors):
        p, _ = prob_vectors
        skl = symmetrized_kl(p, p)
        assert jnp.allclose(skl, 0.0, atol=1e-6)

    def test_non_negative(self, prob_vectors):
        p, q = prob_vectors
        skl = symmetrized_kl(p, q)
        assert jnp.all(skl >= -1e-7)


# ---------------------------------------------------------------------------
# Fisher-Rao distance tests
# ---------------------------------------------------------------------------


class TestFisherRaoDistance:
    """Test Fisher-Rao geodesic distance."""

    def test_fisher_rao_distance_symmetric(self, prob_vectors):
        """d(p, q) should equal d(q, p)."""
        p, q = prob_vectors
        d_pq = fisher_rao_distance(p, q)
        d_qp = fisher_rao_distance(q, p)
        assert jnp.allclose(d_pq, d_qp, atol=1e-6)

    def test_zero_for_identical(self, prob_vectors):
        p, _ = prob_vectors
        d = fisher_rao_distance(p, p)
        assert jnp.allclose(d, 0.0, atol=1e-3)

    def test_non_negative(self, prob_vectors):
        p, q = prob_vectors
        d = fisher_rao_distance(p, q)
        assert jnp.all(d >= -1e-7)

    def test_shape(self, prob_vectors):
        p, q = prob_vectors
        d = fisher_rao_distance(p, q)
        assert d.shape == (3,)


# ---------------------------------------------------------------------------
# JS divergence tests
# ---------------------------------------------------------------------------


class TestJSDivergence:
    """Test Jensen-Shannon divergence."""

    def test_symmetric(self, prob_vectors):
        p, q = prob_vectors
        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)
        assert jnp.allclose(js_pq, js_qp, atol=1e-6)

    def test_bounded_by_log2(self, prob_vectors):
        p, q = prob_vectors
        js = js_divergence(p, q)
        assert jnp.all(js <= jnp.log(2.0) + 1e-6)

    def test_zero_for_identical(self, prob_vectors):
        p, _ = prob_vectors
        js = js_divergence(p, p)
        assert jnp.allclose(js, 0.0, atol=1e-6)

    def test_non_negative(self, prob_vectors):
        p, q = prob_vectors
        js = js_divergence(p, q)
        assert jnp.all(js >= -1e-7)

    def test_shape(self, prob_vectors):
        p, q = prob_vectors
        js = js_divergence(p, q)
        assert js.shape == (3,)


# ---------------------------------------------------------------------------
# Wasserstein on simplex tests
# ---------------------------------------------------------------------------


class TestWassersteinOnSimplex:
    """Test 1-Wasserstein distance on the simplex."""

    def test_zero_for_identical(self, prob_vectors):
        p, _ = prob_vectors
        w = wasserstein_on_simplex(p, p)
        assert jnp.allclose(w, 0.0, atol=1e-6)

    def test_non_negative(self, prob_vectors):
        p, q = prob_vectors
        w = wasserstein_on_simplex(p, q)
        assert jnp.all(w >= -1e-7)

    def test_symmetric(self, prob_vectors):
        p, q = prob_vectors
        w_pq = wasserstein_on_simplex(p, q)
        w_qp = wasserstein_on_simplex(q, p)
        assert jnp.allclose(w_pq, w_qp, atol=1e-6)

    def test_shape(self, prob_vectors):
        p, q = prob_vectors
        w = wasserstein_on_simplex(p, q)
        assert w.shape == (3,)

    def test_known_value(self):
        """Wasserstein between delta(0) and delta(2) on 3 categories."""
        p = jnp.array([[1.0, 0.0, 0.0]])
        q = jnp.array([[0.0, 0.0, 1.0]])
        w = wasserstein_on_simplex(p, q)
        # CDF difference: |1-0| + |1-0| = 2
        assert jnp.allclose(w[0], 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Free energy tests
# ---------------------------------------------------------------------------


class TestFreeEnergy:
    """Test variational free energy on hypergraphs."""

    def test_free_energy_finite(self, prng_key):
        """Free energy output should be finite."""
        n, K, d = 4, 5, 8
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jax.random.normal(prng_key, (n, d))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )
        observations = features
        prior = jnp.ones(K) / K

        fe = free_energy(beliefs, observations, prior, hg)
        assert jnp.isfinite(fe)

    def test_free_energy_scalar(self, prng_key):
        """Free energy should be a scalar."""
        n, K, d = 3, 4, 6
        H = jnp.eye(n)
        features = jnp.ones((n, d))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jnp.ones((n, K)) / K
        observations = features
        prior = jnp.ones(K) / K

        fe = free_energy(beliefs, observations, prior, hg)
        assert fe.shape == ()


# ---------------------------------------------------------------------------
# InfoGeometricDynamics tests
# ---------------------------------------------------------------------------


class TestInfoGeometricDynamics:
    """Test InfoGeometricDynamics module."""

    def test_info_dynamics_shape(self, prng_key):
        """Output shape should match input beliefs shape."""
        n, K = 4, 5
        hidden_dim = 16
        model = InfoGeometricDynamics(
            num_categories=K, hidden_dim=hidden_dim, key=prng_key
        )

        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jnp.ones((n, K))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )
        out = model(jnp.array(0.0), beliefs, hg)
        assert out.shape == (n, K)

    def test_info_dynamics_tangent_sum_zero(self, prng_key):
        """Drift vectors should sum to approximately 0 per node."""
        n, K = 4, 5
        hidden_dim = 16
        model = InfoGeometricDynamics(
            num_categories=K, hidden_dim=hidden_dim, key=prng_key
        )

        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jnp.ones((n, K))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )
        drift = model(jnp.array(0.0), beliefs, hg)

        row_sums = jnp.sum(drift, axis=-1)
        assert jnp.allclose(row_sums, 0.0, atol=1e-5), (
            f"Row sums should be ~0, got {row_sums}"
        )

    def test_finite_output(self, prng_key):
        """All drift values should be finite."""
        n, K = 3, 4
        model = InfoGeometricDynamics(
            num_categories=K, hidden_dim=8, key=prng_key
        )

        H = jnp.eye(n)
        features = jnp.ones((n, K))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jax.nn.softmax(jnp.ones((n, K)), axis=-1)
        out = model(jnp.array(0.0), beliefs, hg)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Belief update tests
# ---------------------------------------------------------------------------


class TestInfoBeliefUpdate:
    """Test belief propagation via information geometry."""

    def test_belief_update_stays_on_simplex(self, prng_key):
        """Beliefs should sum to 1 after update (on the simplex)."""
        n, K, d = 4, 5, 8
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jax.random.normal(prng_key, (n, d))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )
        observations = features

        updated = info_belief_update(
            beliefs, hg, observations, num_steps=10
        )

        # Each row should sum to 1
        row_sums = jnp.sum(updated, axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5), (
            f"Row sums should be 1.0, got {row_sums}"
        )

    def test_shape_preserved(self, prng_key):
        """Output shape should match input beliefs shape."""
        n, K, d = 4, 5, 8
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jax.random.normal(prng_key, (n, d))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )
        updated = info_belief_update(beliefs, hg, features, num_steps=5)
        assert updated.shape == (n, K)

    def test_finite_output(self, prng_key):
        """Updated beliefs should be finite."""
        n, K, d = 3, 4, 6
        H = jnp.eye(n)
        features = jnp.ones((n, d))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jnp.ones((n, K)) / K
        updated = info_belief_update(
            beliefs, hg, features, num_steps=10
        )
        assert jnp.all(jnp.isfinite(updated))

    def test_non_negative(self, prng_key):
        """Updated beliefs should be non-negative (probabilities)."""
        n, K, d = 4, 5, 8
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jax.random.normal(prng_key, (n, d))
        hg = hgx.from_incidence(H, node_features=features)

        beliefs = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )
        updated = info_belief_update(beliefs, hg, features, num_steps=5)
        assert jnp.all(updated >= 0.0)


# ---------------------------------------------------------------------------
# Natural gradient descent tests
# ---------------------------------------------------------------------------


class TestNaturalGradientDescent:
    """Test iterative natural gradient descent."""

    def test_output_on_simplex(self):
        """Output should remain on the probability simplex."""
        p = jax.nn.softmax(jnp.array([
            [1.0, 2.0, 0.5],
            [0.1, 0.3, 0.6],
        ]), axis=-1)

        def loss_fn(p):
            return jnp.sum(p ** 2)

        result = natural_gradient_descent(
            loss_fn, p, lr=0.01, num_steps=10
        )
        row_sums = jnp.sum(result, axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        p = jax.nn.softmax(jnp.ones((3, 4)), axis=-1)

        def loss_fn(p):
            return jnp.sum(p ** 2)

        result = natural_gradient_descent(
            loss_fn, p, lr=0.01, num_steps=5
        )
        assert result.shape == (3, 4)


# ---------------------------------------------------------------------------
# FisherRaoDrift tests
# ---------------------------------------------------------------------------


class TestFisherRaoDrift:
    """Test FisherRaoDrift module."""

    def test_output_shape(self, prng_key):
        """Output should have same shape as input."""
        K = 6
        n = 4
        conv = hgx.UniGCNConv(in_dim=K, out_dim=K, key=prng_key)
        drift = FisherRaoDrift(conv=conv)

        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        y = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )

        args = {"incidence": H}
        out = drift(jnp.array(0.0), y, args)
        assert out.shape == (n, K)

    def test_preserves_simplex(self, prng_key):
        """Drift should be tangent to the simplex (zero mean per row).

        Evolving by a small step should keep outputs close to simplex.
        """
        K = 6
        n = 4
        conv = hgx.UniGCNConv(in_dim=K, out_dim=K, key=prng_key)
        drift = FisherRaoDrift(conv=conv)

        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        y = jax.nn.softmax(
            jax.random.normal(prng_key, (n, K)), axis=-1
        )

        args = {"incidence": H}
        dy = drift(jnp.array(0.0), y, args)

        # Drift should sum to ~0 per row (tangent to simplex)
        row_sums = jnp.sum(dy, axis=-1)
        assert jnp.allclose(row_sums, 0.0, atol=1e-5)

        # Small Euler step should stay close to simplex
        y_next = y + 1e-4 * dy
        row_totals = jnp.sum(y_next, axis=-1)
        assert jnp.allclose(row_totals, 1.0, atol=1e-3)

    def test_finite_output(self, prng_key):
        """Output should be finite."""
        K = 4
        n = 3
        conv = hgx.UniGCNConv(in_dim=K, out_dim=K, key=prng_key)
        drift = FisherRaoDrift(conv=conv)

        H = jnp.eye(n)
        y = jax.nn.softmax(jnp.ones((n, K)), axis=-1)
        args = {"incidence": H}
        out = drift(jnp.array(0.0), y, args)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# FreeEnergyDrift tests
# ---------------------------------------------------------------------------


class TestFreeEnergyDrift:
    """Test FreeEnergyDrift module."""

    def test_output_shape(self, prng_key):
        """Output should match input shape."""
        state_dim = 6
        obs_dim = 6
        n = 4
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(
            in_dim=state_dim, out_dim=obs_dim, key=k1
        )
        drift = FreeEnergyDrift(
            state_dim=state_dim,
            obs_dim=obs_dim,
            conv=conv,
            key=k2,
        )

        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        y = jax.nn.softmax(
            jax.random.normal(prng_key, (n, state_dim)), axis=-1
        )
        args = {"incidence": H}

        out = drift(jnp.array(0.0), y, args)
        assert out.shape == (n, state_dim)

    def test_finite_output(self, prng_key):
        """Output should be finite."""
        state_dim = 4
        obs_dim = 4
        n = 3
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(
            in_dim=state_dim, out_dim=obs_dim, key=k1
        )
        drift = FreeEnergyDrift(
            state_dim=state_dim,
            obs_dim=obs_dim,
            conv=conv,
            key=k2,
        )

        H = jnp.eye(n)
        y = jax.nn.softmax(jnp.ones((n, state_dim)), axis=-1)
        args = {"incidence": H}

        out = drift(jnp.array(0.0), y, args)
        assert jnp.all(jnp.isfinite(out))

    def test_tangent_to_simplex(self, prng_key):
        """Drift should be tangent to the simplex."""
        state_dim = 4
        obs_dim = 4
        n = 3
        k1, k2 = jax.random.split(prng_key)
        conv = hgx.UniGCNConv(
            in_dim=state_dim, out_dim=obs_dim, key=k1
        )
        drift = FreeEnergyDrift(
            state_dim=state_dim,
            obs_dim=obs_dim,
            conv=conv,
            key=k2,
        )

        H = jnp.eye(n)
        y = jax.nn.softmax(
            jax.random.normal(k1, (n, state_dim)), axis=-1
        )
        args = {"incidence": H}

        dy = drift(jnp.array(0.0), y, args)
        row_sums = jnp.sum(dy, axis=-1)
        assert jnp.allclose(row_sums, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# InfoGeometricODE tests
# ---------------------------------------------------------------------------


class TestInfoGeometricODE:
    """Test the complete InfoGeometricODE model."""

    def test_output_is_hypergraph(self, small_hypergraph, prng_key):
        """Output should be a Hypergraph with correct feature shape."""
        k1, k2 = jax.random.split(prng_key)
        num_states = 6
        conv = hgx.UniGCNConv(
            in_dim=num_states, out_dim=num_states, key=k1
        )
        model = InfoGeometricODE(
            feature_dim=8, num_states=num_states, conv=conv, key=k2
        )
        hg_out = model(small_hypergraph, t0=0.0, t1=1.0)
        assert isinstance(hg_out, hgx.Hypergraph)
        assert hg_out.node_features.shape == (4, 8)

    def test_structure_preserved(self, small_hypergraph, prng_key):
        """Incidence matrix should be preserved."""
        k1, k2 = jax.random.split(prng_key)
        num_states = 6
        conv = hgx.UniGCNConv(
            in_dim=num_states, out_dim=num_states, key=k1
        )
        model = InfoGeometricODE(
            feature_dim=8, num_states=num_states, conv=conv, key=k2
        )
        hg_out = model(small_hypergraph, t0=0.0, t1=1.0)
        assert jnp.array_equal(
            hg_out.incidence, small_hypergraph.incidence
        )

    def test_finite_output(self, small_hypergraph, prng_key):
        """All output features should be finite."""
        k1, k2 = jax.random.split(prng_key)
        num_states = 6
        conv = hgx.UniGCNConv(
            in_dim=num_states, out_dim=num_states, key=k1
        )
        model = InfoGeometricODE(
            feature_dim=8, num_states=num_states, conv=conv, key=k2
        )
        hg_out = model(small_hypergraph, t0=0.0, t1=1.0)
        assert jnp.all(jnp.isfinite(hg_out.node_features))
