"""Tests for Lorentz (hyperboloid) model hypergraph convolution."""

from __future__ import annotations

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hgx._conv._lorentz import (
    einstein_midpoint,
    exp_map_0,
    log_map_0,
    lorentz_inner,
    lorentz_norm_sq,
    LorentzHypergraphConv,
    project_to_hyperboloid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lorentz_hg(n: int, m: int, dim: int, c: float = 1.0, *, key):
    """Build a Hypergraph with features on the hyperboloid H^dim_c.

    ``dim`` is the *spatial* dimension, so ambient dim = dim + 1.
    """
    k1, k2 = jax.random.split(key)
    # Random spatial coordinates, then project onto hyperboloid
    spatial = 0.3 * jax.random.normal(k1, (n, dim))
    x0 = jnp.sqrt(1.0 / c + jnp.sum(spatial ** 2, axis=-1, keepdims=True))
    features = jnp.concatenate([x0, spatial], axis=-1)  # (n, dim+1)

    # Random incidence (at least 2 members per edge)
    H_raw = jax.random.bernoulli(k2, 0.5, (n, m)).astype(jnp.float32)
    # Ensure each edge has at least 2 members
    H = H_raw.at[:2, :].set(1.0)

    return hgx.Hypergraph(
        node_features=features,
        incidence=H,
        geometry="lorentz",
    )


def _on_hyperboloid(x, c, atol=1e-4):
    """Check <x, x>_L ≈ -1/c for all rows."""
    norms = lorentz_norm_sq(x)
    expected = -1.0 / c
    return np.allclose(np.asarray(norms), expected, atol=atol)


# ---------------------------------------------------------------------------
# Lorentz primitive tests
# ---------------------------------------------------------------------------

class TestLorentzPrimitives:
    def test_lorentz_inner_signature(self):
        x = jnp.array([2.0, 1.0, 0.5])
        y = jnp.array([3.0, 0.5, 1.0])
        # -2*3 + 1*0.5 + 0.5*1 = -6 + 0.5 + 0.5 = -5.0
        assert jnp.isclose(lorentz_inner(x, y), -5.0)

    def test_lorentz_inner_batched(self):
        x = jnp.ones((4, 3))
        y = jnp.ones((4, 3))
        result = lorentz_inner(x, y)
        # -1 + 1 + 1 = 1 for each row
        np.testing.assert_allclose(result, 1.0, atol=1e-6)
        assert result.shape == (4,)

    def test_project_to_hyperboloid(self):
        c = jnp.array(1.0)
        x = jnp.array([[0.0, 0.3, 0.4],  # spatial = [0.3, 0.4]
                        [0.0, 0.0, 0.0]])  # origin-like
        proj = project_to_hyperboloid(x, c)
        assert _on_hyperboloid(proj, float(c))

    def test_project_different_curvature(self):
        c = jnp.array(2.0)
        x = jnp.array([[0.0, 0.5, 0.5]])
        proj = project_to_hyperboloid(x, c)
        norms = lorentz_norm_sq(proj)
        np.testing.assert_allclose(np.asarray(norms), -1.0 / 2.0, atol=1e-5)

    def test_exp_log_roundtrip(self):
        """exp_0(log_0(x)) ≈ x for points on the hyperboloid."""
        c = jnp.array(1.0)
        # Create a point on the hyperboloid
        spatial = jnp.array([[0.3, -0.2, 0.1]])
        x = project_to_hyperboloid(
            jnp.concatenate([jnp.zeros((1, 1)), spatial], axis=-1), c
        )
        v = log_map_0(x, c)
        x_rec = exp_map_0(v, c)
        np.testing.assert_allclose(np.asarray(x_rec), np.asarray(x), atol=1e-4)

    def test_exp_map_origin(self):
        """exp_0(0) = origin on hyperboloid."""
        c = jnp.array(1.0)
        v = jnp.zeros((1, 4))
        x = exp_map_0(v, c)
        # Should be approximately (1/sqrt(c), 0, 0, 0) = (1, 0, 0, 0)
        np.testing.assert_allclose(np.asarray(x[0, 0]), 1.0, atol=1e-3)
        np.testing.assert_allclose(np.asarray(x[0, 1:]), 0.0, atol=1e-5)

    def test_log_map_origin(self):
        """log_0(origin) ≈ 0."""
        c = jnp.array(1.0)
        origin = jnp.array([[1.0 / jnp.sqrt(c), 0.0, 0.0, 0.0]])
        # Very close to origin — log should be near zero
        v = log_map_0(origin, c)
        np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-3)


class TestEinsteinMidpoint:
    def test_single_point(self):
        c = jnp.array(1.0)
        x = project_to_hyperboloid(jnp.array([[0.0, 0.5, -0.3]]), c)
        mid = einstein_midpoint(x, c=c)
        np.testing.assert_allclose(np.asarray(mid), np.asarray(x[0]), atol=1e-4)

    def test_two_symmetric_points(self):
        """Midpoint of symmetric points should be near origin."""
        c = jnp.array(1.0)
        p1 = project_to_hyperboloid(jnp.array([[0.0, 0.5, 0.0]]), c)
        p2 = project_to_hyperboloid(jnp.array([[0.0, -0.5, 0.0]]), c)
        x = jnp.concatenate([p1, p2], axis=0)
        mid = einstein_midpoint(x, c=c)
        # Midpoint should have near-zero spatial components
        np.testing.assert_allclose(np.asarray(mid[1:]), 0.0, atol=0.05)
        # And lie on the hyperboloid
        assert _on_hyperboloid(mid[None], float(c), atol=1e-3)

    def test_midpoint_on_hyperboloid(self):
        """Midpoint of multiple points should lie on the hyperboloid."""
        c = jnp.array(1.0)
        key = jax.random.PRNGKey(42)
        spatial = 0.3 * jax.random.normal(key, (5, 3))
        x = project_to_hyperboloid(
            jnp.concatenate([jnp.zeros((5, 1)), spatial], axis=-1), c
        )
        mid = einstein_midpoint(x, c=c)
        assert _on_hyperboloid(mid[None], float(c), atol=1e-3)

    def test_weighted_midpoint(self):
        """With one weight = 0, midpoint should be the other point."""
        c = jnp.array(1.0)
        p1 = project_to_hyperboloid(jnp.array([[0.0, 0.5, 0.0]]), c)
        p2 = project_to_hyperboloid(jnp.array([[0.0, -0.3, 0.2]]), c)
        x = jnp.concatenate([p1, p2], axis=0)
        # Weight 0 for p2 -> should recover p1
        mid = einstein_midpoint(x, weights=jnp.array([1.0, 0.0]), c=c)
        np.testing.assert_allclose(np.asarray(mid), np.asarray(p1[0]), atol=1e-3)


# ---------------------------------------------------------------------------
# LorentzHypergraphConv tests
# ---------------------------------------------------------------------------

class TestLorentzHypergraphConv:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def lorentz_hg(self, key):
        return _make_lorentz_hg(n=6, m=3, dim=4, c=1.0, key=key)

    def test_output_shape(self, lorentz_hg, key):
        conv = LorentzHypergraphConv(in_dim=5, out_dim=5, key=key)
        out = conv(lorentz_hg)
        assert out.shape == (6, 5)

    def test_output_shape_different_dims(self, lorentz_hg, key):
        conv = LorentzHypergraphConv(in_dim=5, out_dim=4, key=key)
        out = conv(lorentz_hg)
        assert out.shape == (6, 4)

    def test_output_on_hyperboloid(self, lorentz_hg, key):
        """Output features must lie on the hyperboloid <x,x>_L = -1/c."""
        conv = LorentzHypergraphConv(in_dim=5, out_dim=5, key=key)
        out = conv(lorentz_hg)
        c = float(conv.c)
        assert _on_hyperboloid(out, c, atol=1e-3), (
            f"Output not on hyperboloid. Lorentz norms: {lorentz_norm_sq(out)}, "
            f"expected {-1.0 / c}"
        )

    def test_output_on_hyperboloid_different_curvature(self, key):
        """Output on hyperboloid with non-unit curvature."""
        hg = _make_lorentz_hg(n=4, m=2, dim=3, c=2.0, key=key)
        conv = LorentzHypergraphConv(in_dim=4, out_dim=4, init_curvature=2.0, key=key)
        out = conv(hg)
        c = float(conv.c)
        assert _on_hyperboloid(out, c, atol=1e-3)

    def test_learnable_curvature(self, key):
        """Curvature parameter should be differentiable."""
        hg = _make_lorentz_hg(n=4, m=2, dim=3, c=1.0, key=key)
        conv = LorentzHypergraphConv(in_dim=4, out_dim=4, key=key)

        def loss_fn(model):
            out = model(hg)
            return jnp.sum(out ** 2)

        grads = eqx.filter_grad(loss_fn)(conv)
        # log_c should have a non-trivial gradient
        assert grads.log_c is not None
        assert not jnp.isnan(grads.log_c)

    def test_jit_compatible(self, lorentz_hg, key):
        conv = LorentzHypergraphConv(in_dim=5, out_dim=5, key=key)
        jit_conv = eqx.filter_jit(conv)
        out = jit_conv(lorentz_hg)
        assert out.shape == (6, 5)
        assert not jnp.any(jnp.isnan(out))

    def test_gradient_flow(self, lorentz_hg, key):
        """Gradients should flow through the layer without NaNs."""
        conv = LorentzHypergraphConv(in_dim=5, out_dim=5, key=key)

        def loss_fn(model):
            out = model(lorentz_hg)
            return jnp.mean(out ** 2)

        grads = eqx.filter_grad(loss_fn)(conv)
        # Check linear weight grads exist and are finite
        assert grads.linear.weight is not None
        assert jnp.all(jnp.isfinite(grads.linear.weight))

    def test_gradient_flow_through_curvature(self, lorentz_hg, key):
        """Curvature gradient should be finite."""
        conv = LorentzHypergraphConv(in_dim=5, out_dim=5, key=key)

        def loss_fn(model):
            out = model(lorentz_hg)
            return jnp.mean(out)

        grads = eqx.filter_grad(loss_fn)(conv)
        assert jnp.isfinite(grads.log_c)

    def test_non_lorentz_geometry_projects(self, key):
        """When geometry != "lorentz", features should be projected first."""
        # Euclidean features — not on hyperboloid
        hg = hgx.from_edge_list(
            [(0, 1, 2), (1, 2, 3)],
            num_nodes=4,
            node_features=jnp.ones((4, 4)),
        )
        conv = LorentzHypergraphConv(in_dim=4, out_dim=4, key=key)
        out = conv(hg)
        # Should still produce valid output on hyperboloid
        c = float(conv.c)
        assert _on_hyperboloid(out, c, atol=1e-3)

    def test_lorentz_geometry_activates(self, key):
        """With geometry="lorentz", no projection should occur."""
        hg = _make_lorentz_hg(n=4, m=2, dim=3, c=1.0, key=key)
        assert hg.geometry == "lorentz"
        conv = LorentzHypergraphConv(in_dim=4, out_dim=4, key=key)
        out = conv(hg)
        assert out.shape == (4, 4)

    def test_node_mask_respected(self, key):
        """Masked nodes should produce zero output."""
        hg_base = _make_lorentz_hg(n=4, m=2, dim=3, c=1.0, key=key)
        mask = jnp.array([True, True, False, True])
        hg = hgx.Hypergraph(
            node_features=hg_base.node_features,
            incidence=hg_base.incidence,
            node_mask=mask,
            geometry="lorentz",
        )
        conv = LorentzHypergraphConv(in_dim=4, out_dim=4, key=key)
        out = conv(hg)
        # Node 2 is masked — output should be zero
        np.testing.assert_allclose(np.asarray(out[2]), 0.0, atol=1e-6)

    def test_invalid_dims_raises(self, key):
        with pytest.raises(ValueError, match="in_dim and out_dim must be >= 2"):
            LorentzHypergraphConv(in_dim=1, out_dim=3, key=key)

    def test_is_abstract_conv_subclass(self, key):
        conv = LorentzHypergraphConv(in_dim=4, out_dim=4, key=key)
        assert isinstance(conv, hgx.AbstractHypergraphConv)

    def test_tiny_fixture(self, key):
        """Works on the shared tiny_hypergraph fixture (2D features -> ambient 2)."""
        hg = hgx.from_edge_list(
            [(0, 1, 2), (1, 2, 3)],
            num_nodes=4,
            node_features=jnp.ones((4, 3)),
        )
        conv = LorentzHypergraphConv(in_dim=3, out_dim=3, key=key)
        out = conv(hg)
        assert out.shape == (4, 3)
        c = float(conv.c)
        assert _on_hyperboloid(out, c, atol=1e-3)
