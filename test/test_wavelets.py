"""Tests for spectral wavelet transforms on hypergraphs."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
from hgx._transforms import hypergraph_laplacian
from hgx._wavelets import (
    _chebyshev_filter,
    cheeger_constant_bound,
    hypergraph_scattering,
    hypergraph_wavelet_transform,
    HypergraphWaveletLayer,
    spectral_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hg(n=8, m=3, d=4, *, key):
    """Create a random hypergraph with n nodes, m edges, d feature dim."""
    k1, k2 = jax.random.split(key)
    H = jax.random.bernoulli(k1, 0.4, shape=(n, m)).astype(jnp.float32)
    # Ensure every edge has at least one node and every node at least one edge
    H = H.at[jnp.arange(min(n, m)), jnp.arange(min(n, m))].set(1.0)
    features = jax.random.normal(k2, (n, d))
    return hgx.from_incidence(H, node_features=features)


# ---------------------------------------------------------------------------
# hypergraph_wavelet_transform
# ---------------------------------------------------------------------------


class TestHypergraphWaveletTransform:
    """Tests for the spectral wavelet transform."""

    def test_output_shape(self, prng_key):
        """Wavelet coefficients should have shape (num_scales, n, d)."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        scales = [0.5, 1.0, 2.0, 4.0]
        coeffs = hypergraph_wavelet_transform(hg, scales)
        assert coeffs.shape == (4, 6, 4)

    def test_output_shape_single_scale(self, prng_key):
        """Single scale should give shape (1, n, d)."""
        hg = _make_hg(n=5, m=2, d=3, key=prng_key)
        coeffs = hypergraph_wavelet_transform(hg, [1.0])
        assert coeffs.shape == (1, 5, 3)

    def test_heat_kernel_small_scale_near_identity(self, prng_key):
        """Heat kernel at s -> 0 should approximate identity (no smoothing)."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        # At very small scale, exp(-s*lambda) ~ 1 for all lambda, so W ~ I
        coeffs = hypergraph_wavelet_transform(hg, [1e-6], kernel="heat")
        # The filtered signal should be close to the original features
        assert jnp.allclose(coeffs[0], hg.node_features, atol=1e-3)

    def test_heat_kernel_large_scale_smoothed(self, prng_key):
        """Heat kernel at large s should heavily smooth the signal."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        coeffs_small = hypergraph_wavelet_transform(hg, [0.1], kernel="heat")
        coeffs_large = hypergraph_wavelet_transform(hg, [100.0], kernel="heat")
        # Large-scale coefficients should have smaller variance (more smoothed)
        var_small = jnp.var(coeffs_small)
        var_large = jnp.var(coeffs_large)
        assert var_large < var_small

    def test_mexican_hat_kernel(self, prng_key):
        """Mexican hat kernel should produce finite coefficients."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        coeffs = hypergraph_wavelet_transform(hg, [0.5, 1.0, 2.0], kernel="mexican_hat")
        assert jnp.all(jnp.isfinite(coeffs))

    def test_meyer_kernel(self, prng_key):
        """Meyer kernel should produce finite coefficients."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        coeffs = hypergraph_wavelet_transform(hg, [0.5, 1.0, 2.0], kernel="meyer")
        assert jnp.all(jnp.isfinite(coeffs))

    def test_coefficients_finite(self, tiny_hypergraph):
        """All wavelet coefficients should be finite."""
        coeffs = hypergraph_wavelet_transform(
            tiny_hypergraph, [0.1, 1.0, 10.0]
        )
        assert jnp.all(jnp.isfinite(coeffs))


# ---------------------------------------------------------------------------
# HypergraphWaveletLayer (ChebNet)
# ---------------------------------------------------------------------------


class TestHypergraphWaveletLayer:
    """Tests for the learnable Chebyshev spectral filter."""

    def test_output_shape(self, prng_key):
        """Output should have shape (n, out_dim)."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        layer = HypergraphWaveletLayer(in_dim=4, out_dim=8, K=5, key=k1)
        out = layer(hg)
        assert out.shape == (6, 8)

    def test_jit_compatible(self, prng_key):
        """Layer should work under eqx.filter_jit."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        layer = HypergraphWaveletLayer(in_dim=4, out_dim=8, K=3, key=k1)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out = forward(layer, hg)
        assert out.shape == (6, 8)
        assert jnp.all(jnp.isfinite(out))

    def test_differentiable(self, prng_key):
        """Gradients should flow through the Chebyshev filter."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        layer = HypergraphWaveletLayer(in_dim=4, out_dim=8, K=5, key=k1)

        @eqx.filter_grad
        def grad_fn(model):
            return jnp.sum(model(hg))

        grads = grad_fn(layer)
        # Chebyshev coefficients should have non-zero grads
        assert not jnp.allclose(grads.chebyshev_coeffs, 0.0)
        # Linear layer should also have gradients
        assert not jnp.allclose(grads.linear.weight, 0.0)

    def test_chebyshev_matches_exact_filter(self, prng_key):
        """Chebyshev polynomial filter should approximate exact spectral filter.

        For a polynomial filter h(lambda) = sum_k a_k T_k(lambda_tilde),
        the Chebyshev approximation applied via the recurrence should match
        the result of computing the filter spectrally via eigendecomposition.
        """
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        L = hypergraph_laplacian(hg, normalized=True)
        x = hg.node_features

        # Use simple coefficients: a_0 = 1, a_1 = 0.5, rest zero
        coeffs = jnp.array([1.0, 0.5, 0.0])

        # Method 1: Chebyshev recurrence
        result_cheb = _chebyshev_filter(L, x, coeffs)

        # Method 2: Exact spectral computation
        eigenvalues, U = jnp.linalg.eigh(L)
        # Rescale to [-1, 1]
        eigenvalues_tilde = eigenvalues - 1.0
        # T_0(x) = 1, T_1(x) = x
        h = coeffs[0] * jnp.ones_like(eigenvalues_tilde)
        h = h + coeffs[1] * eigenvalues_tilde
        # T_2(x) = 2x^2 - 1
        h = h + coeffs[2] * (2.0 * eigenvalues_tilde**2 - 1.0)
        result_exact = U @ jnp.diag(h) @ U.T @ x

        assert jnp.allclose(result_cheb, result_exact, atol=1e-4)

    def test_different_polynomial_orders(self, prng_key):
        """Layer should work with different polynomial orders K."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        for K in [1, 2, 5, 10]:
            k1, prng_key = jax.random.split(prng_key)
            layer = HypergraphWaveletLayer(in_dim=4, out_dim=4, K=K, key=k1)
            out = layer(hg)
            assert out.shape == (6, 4)
            assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# hypergraph_scattering
# ---------------------------------------------------------------------------


class TestHypergraphScattering:
    """Tests for the wavelet scattering transform."""

    def test_fixed_size_output(self, prng_key):
        """Scattering output should be a fixed-size 1-d vector."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        scales = [1.0, 2.0]
        out = hypergraph_scattering(hg, scales, num_layers=2)
        assert out.ndim == 1
        # Size = d + S*d + S^2*d = 4 + 2*4 + 4*4 = 4 + 8 + 16 = 28
        assert out.shape == (28,)

    def test_different_graph_sizes_same_feature_size(self, prng_key):
        """Graphs of different sizes should produce same-size scattering vectors."""
        k1, k2 = jax.random.split(prng_key)
        hg_small = _make_hg(n=5, m=2, d=3, key=k1)
        hg_large = _make_hg(n=20, m=6, d=3, key=k2)
        scales = [0.5, 1.0, 2.0]

        out_small = hypergraph_scattering(hg_small, scales, num_layers=1)
        out_large = hypergraph_scattering(hg_large, scales, num_layers=1)
        assert out_small.shape == out_large.shape

    def test_output_finite(self, prng_key):
        """All scattering coefficients should be finite."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        out = hypergraph_scattering(hg, [1.0, 2.0], num_layers=2)
        assert jnp.all(jnp.isfinite(out))

    def test_single_layer(self, prng_key):
        """Single-layer scattering should work."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        scales = [1.0, 2.0]
        out = hypergraph_scattering(hg, scales, num_layers=1)
        # Size = d + S*d = 4 + 2*4 = 12
        assert out.shape == (12,)


# ---------------------------------------------------------------------------
# spectral_features
# ---------------------------------------------------------------------------


class TestSpectralFeatures:
    """Tests for spectral feature extraction."""

    def test_feature_length(self, prng_key):
        """Feature vector should have length num_eigvals + 3."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        feats = spectral_features(hg, num_eigvals=10)
        assert feats.shape == (13,)

    def test_feature_length_custom(self, prng_key):
        """Custom num_eigvals should produce correct size."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        feats = spectral_features(hg, num_eigvals=5)
        assert feats.shape == (8,)

    def test_finite_values(self, prng_key):
        """All spectral features should be finite."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        feats = spectral_features(hg)
        assert jnp.all(jnp.isfinite(feats))

    def test_smallest_eigenvalue_near_zero(self, tiny_hypergraph):
        """Smallest eigenvalue of normalized Laplacian should be ~ 0."""
        feats = spectral_features(tiny_hypergraph, num_eigvals=4)
        # First eigenvalue (index 0) should be close to zero
        assert jnp.abs(feats[0]) < 1e-5

    def test_algebraic_connectivity_nonneg(self, prng_key):
        """Algebraic connectivity (lambda_2) should be non-negative."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        feats = spectral_features(hg)
        # algebraic_connectivity is at index num_eigvals + 1
        assert feats[11] >= -1e-6  # index 10+1 = 11


# ---------------------------------------------------------------------------
# cheeger_constant_bound
# ---------------------------------------------------------------------------


class TestCheegerConstantBound:
    """Tests for the Cheeger constant bounds."""

    def test_bounds_ordering(self, prng_key):
        """Lower bound should not exceed upper bound."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        lower, upper = cheeger_constant_bound(hg)
        assert lower <= upper + 1e-6

    def test_bounds_nonneg(self, prng_key):
        """Both bounds should be non-negative."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        lower, upper = cheeger_constant_bound(hg)
        assert lower >= -1e-6
        assert upper >= -1e-6

    def test_cheeger_inequality_consistent(self, prng_key):
        """Bounds should satisfy h^2/2 <= lambda_2 <= 2h.

        The Cheeger inequality states:
            lambda_2 / 2 <= h <= sqrt(2 * lambda_2)

        Squaring the lower bound: h >= lambda_2/2 => h^2 >= lambda_2^2/4
        For the upper bound: h <= sqrt(2*lambda_2) => h^2 <= 2*lambda_2

        We verify the bounds are self-consistent.
        """
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        lower, upper = cheeger_constant_bound(hg)

        L = hypergraph_laplacian(hg, normalized=True)
        eigenvalues = jnp.sort(jnp.linalg.eigvalsh(L))
        lambda_2 = float(eigenvalues[1])

        # lower_bound = lambda_2 / 2, upper_bound = sqrt(2 * lambda_2)
        assert abs(lower - lambda_2 / 2.0) < 1e-5
        assert abs(upper - float(jnp.sqrt(2.0 * max(lambda_2, 0.0)))) < 1e-5

        # The inequality: lower^2 * 2 <= lambda_2 if h = lower
        # Actually: h >= lambda_2/2, and h^2 <= 2*lambda_2
        # So lower^2 <= 2*lambda_2 should hold
        assert lower**2 <= 2.0 * lambda_2 + 1e-5

    def test_tiny_graph(self, tiny_hypergraph):
        """Should work on the tiny fixture."""
        lower, upper = cheeger_constant_bound(tiny_hypergraph)
        assert lower >= -1e-6
        assert upper >= lower - 1e-6
