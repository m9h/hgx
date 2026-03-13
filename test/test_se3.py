"""Tests for SE(3)-equivariant hypergraph convolution."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hgx._hypergraph import from_incidence, Hypergraph


try:
    import e3nn_jax as e3nn
    from hgx._conv._se3 import SE3HypergraphConv

    _HAS_E3NN = True
except ImportError:
    _HAS_E3NN = False

requires_e3nn = pytest.mark.skipif(not _HAS_E3NN, reason="e3nn-jax not installed")


def _random_rotation(seed: int = 0) -> jnp.ndarray:
    """Generate a random SO(3) rotation matrix via QR decomposition."""
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (3, 3))
    Q, R = jnp.linalg.qr(A)
    # Ensure proper rotation (det=+1)
    Q = Q * jnp.sign(jnp.linalg.det(Q))
    return Q


def _make_geometric_hypergraph(
    key: jax.Array,
    n: int = 6,
    m: int = 3,
    d: int = 4,
) -> Hypergraph:
    """Create a random hypergraph with 3D positions."""
    k1, k2, k3 = jax.random.split(key, 3)
    H = jax.random.bernoulli(k1, 0.5, (n, m)).astype(jnp.float32)
    # Ensure every edge has at least one node and every node at least one edge
    H = H.at[0, :].set(1.0)
    H = H.at[:, 0].set(1.0)
    feats = jax.random.normal(k2, (n, d))
    pos = jax.random.normal(k3, (n, 3))
    return from_incidence(H, node_features=feats, positions=pos, geometry="euclidean")


# ---------------------------------------------------------------------------
# Forward pass shapes
# ---------------------------------------------------------------------------


@requires_e3nn
class TestSE3ForwardShapes:
    """Test output shapes for various irrep configurations."""

    def test_scalar_plus_vector(self):
        key = jax.random.PRNGKey(0)
        hg = _make_geometric_hypergraph(key, n=5, m=2, d=3)
        conv = SE3HypergraphConv(
            in_dim=3, out_irreps="0e + 1o", key=key,
        )
        out = conv(hg)
        # 0e -> 1, 1o -> 3, total 4
        assert out.shape == (5, 4)

    def test_pure_scalar_output(self):
        key = jax.random.PRNGKey(1)
        hg = _make_geometric_hypergraph(key, n=4, m=2, d=2)
        conv = SE3HypergraphConv(
            in_dim=2, out_irreps="3x0e", key=key,
        )
        out = conv(hg)
        assert out.shape == (4, 3)

    def test_higher_order_irreps(self):
        key = jax.random.PRNGKey(2)
        hg = _make_geometric_hypergraph(key, n=6, m=3, d=4)
        conv = SE3HypergraphConv(
            in_dim=4,
            out_irreps="2x0e + 1o + 2e",
            hidden_irreps="2x0e + 1o",
            sh_lmax=2,
            key=key,
        )
        out = conv(hg)
        expected_dim = e3nn.Irreps("2x0e + 1o + 2e").dim
        assert out.shape == (6, expected_dim)

    def test_multiple_multiplicities(self):
        key = jax.random.PRNGKey(3)
        hg = _make_geometric_hypergraph(key, n=4, m=2, d=5)
        conv = SE3HypergraphConv(
            in_dim=5,
            out_irreps="4x0e + 2x1o",
            hidden_irreps="3x0e + 1o",
            key=key,
        )
        out = conv(hg)
        expected_dim = e3nn.Irreps("4x0e + 2x1o").dim
        assert out.shape == (4, expected_dim)

    def test_dtype_float32(self):
        key = jax.random.PRNGKey(4)
        hg = _make_geometric_hypergraph(key)
        conv = SE3HypergraphConv(in_dim=4, key=key)
        out = conv(hg)
        assert out.dtype == jnp.float32


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


@requires_e3nn
class TestSE3JIT:
    """Test that the layer works under jax.jit."""

    def test_jit_forward(self):
        key = jax.random.PRNGKey(10)
        hg = _make_geometric_hypergraph(key)
        conv = SE3HypergraphConv(in_dim=4, key=key)

        @jax.jit
        def forward(conv, hg):
            return conv(hg)

        out_jit = forward(conv, hg)
        out_eager = conv(hg)
        assert jnp.allclose(out_jit, out_eager, atol=1e-6)

    def test_jit_grad(self):
        """Gradients flow through the JIT-compiled forward pass."""
        key = jax.random.PRNGKey(11)
        hg = _make_geometric_hypergraph(key, n=4, m=2, d=3)
        conv = SE3HypergraphConv(in_dim=3, key=key)

        import equinox as eqx

        @eqx.filter_jit
        @eqx.filter_grad
        def grad_fn(conv):
            return jnp.sum(conv(hg) ** 2)

        grads = grad_fn(conv)
        # Check that at least some gradient is nonzero
        flat = jax.tree.leaves(grads)
        has_nonzero = any(
            jnp.any(jnp.abs(g) > 0)
            for g in flat
            if isinstance(g, jnp.ndarray)
        )
        assert has_nonzero


# ---------------------------------------------------------------------------
# Rotational equivariance
# ---------------------------------------------------------------------------


@requires_e3nn
class TestSE3Equivariance:
    """Test SE(3) equivariance: rotating positions should rotate outputs."""

    def test_scalar_invariance(self):
        """Scalar (0e) output components should be invariant under rotation."""
        key = jax.random.PRNGKey(20)
        hg = _make_geometric_hypergraph(key, n=5, m=3, d=4)
        conv = SE3HypergraphConv(
            in_dim=4, out_irreps="2x0e", hidden_irreps="2x0e + 1o", key=key,
        )

        out_orig = conv(hg)

        R = _random_rotation(seed=7)
        pos_rot = hg.positions @ R.T
        hg_rot = from_incidence(
            hg.incidence, node_features=hg.node_features,
            positions=pos_rot, geometry="euclidean",
        )
        out_rot = conv(hg_rot)

        # Scalar outputs should be identical
        assert jnp.allclose(out_orig, out_rot, atol=1e-5), (
            f"Max scalar diff: {float(jnp.max(jnp.abs(out_orig - out_rot)))}"
        )

    def test_vector_equivariance(self):
        """Vector (1o) output components should rotate with positions."""
        key = jax.random.PRNGKey(21)
        hg = _make_geometric_hypergraph(key, n=5, m=3, d=4)
        conv = SE3HypergraphConv(
            in_dim=4, out_irreps="1o", hidden_irreps="2x0e + 1o", key=key,
        )

        out_orig = conv(hg)  # (n, 3)

        R = _random_rotation(seed=8)
        pos_rot = hg.positions @ R.T
        hg_rot = from_incidence(
            hg.incidence, node_features=hg.node_features,
            positions=pos_rot, geometry="euclidean",
        )
        out_rot = conv(hg_rot)

        # Rotating original output should match output from rotated input
        out_orig_rotated = out_orig @ R.T
        assert jnp.allclose(out_rot, out_orig_rotated, atol=1e-5), (
            f"Max equivariance error: "
            f"{float(jnp.max(jnp.abs(out_rot - out_orig_rotated)))}"
        )

    def test_mixed_irreps_equivariance(self):
        """Full equivariance test with mixed scalar + vector output."""
        key = jax.random.PRNGKey(22)
        hg = _make_geometric_hypergraph(key, n=6, m=3, d=3)
        conv = SE3HypergraphConv(
            in_dim=3, out_irreps="0e + 1o",
            hidden_irreps="2x0e + 1o", key=key,
        )

        out_orig = conv(hg)

        R = _random_rotation(seed=9)
        pos_rot = hg.positions @ R.T
        hg_rot = from_incidence(
            hg.incidence, node_features=hg.node_features,
            positions=pos_rot, geometry="euclidean",
        )
        out_rot = conv(hg_rot)

        # Use e3nn to rotate the original output
        out_irreps = e3nn.IrrepsArray("0e + 1o", out_orig)
        out_expected = out_irreps.transform_by_matrix(R)

        assert jnp.allclose(out_rot, out_expected.array, atol=1e-5), (
            f"Max equivariance error: "
            f"{float(jnp.max(jnp.abs(out_rot - out_expected.array)))}"
        )

    def test_translation_invariance(self):
        """Output should be invariant to global translation of positions."""
        key = jax.random.PRNGKey(23)
        hg = _make_geometric_hypergraph(key, n=5, m=2, d=3)
        conv = SE3HypergraphConv(
            in_dim=3, out_irreps="0e + 1o", key=key,
        )

        out_orig = conv(hg)

        # Translate all positions
        shift = jnp.array([10.0, -5.0, 3.0])
        pos_shifted = hg.positions + shift[None, :]
        hg_shifted = from_incidence(
            hg.incidence, node_features=hg.node_features,
            positions=pos_shifted, geometry="euclidean",
        )
        out_shifted = conv(hg_shifted)

        assert jnp.allclose(out_orig, out_shifted, atol=1e-5), (
            f"Max translation error: {float(jnp.max(jnp.abs(out_orig - out_shifted)))}"
        )

    def test_equivariance_higher_order(self):
        """Equivariance with l=2 spherical harmonics and 2e output."""
        key = jax.random.PRNGKey(24)
        hg = _make_geometric_hypergraph(key, n=5, m=2, d=3)
        conv = SE3HypergraphConv(
            in_dim=3,
            out_irreps="0e + 1o + 2e",
            hidden_irreps="2x0e + 1o",
            sh_lmax=2,
            key=key,
        )

        out_orig = conv(hg)

        R = _random_rotation(seed=10)
        pos_rot = hg.positions @ R.T
        hg_rot = from_incidence(
            hg.incidence, node_features=hg.node_features,
            positions=pos_rot, geometry="euclidean",
        )
        out_rot = conv(hg_rot)

        out_irreps = e3nn.IrrepsArray("0e + 1o + 2e", out_orig)
        out_expected = out_irreps.transform_by_matrix(R)

        assert jnp.allclose(out_rot, out_expected.array, atol=1e-5), (
            f"Max equivariance error: "
            f"{float(jnp.max(jnp.abs(out_rot - out_expected.array)))}"
        )
