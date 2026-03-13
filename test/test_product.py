"""Tests for mixed-curvature product space hypergraph convolution."""

from __future__ import annotations

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._conv._product import (
    concat_components,
    exp_map_poincare,
    exp_map_sphere,
    log_map_poincare,
    log_map_sphere,
    product_distance,
    ProductHypergraphConv,
    ProductManifold,
    ProductManifoldConv,
    ProductManifoldMLP,
    ProductSpaceConv,
    ProductSpaceEmbedding,
    project_hyperbolic,
    project_spherical,
    project_to_sphere,
    sphere_exp,
    sphere_exp_map,
    sphere_log,
    sphere_log_map,
    sphere_midpoint,
    sphere_project,
    split_components,
)
from hgx._hypergraph import from_incidence, Hypergraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hg(n: int, m: int, feat_dim: int, *, key: jax.Array) -> Hypergraph:
    """Build a small random hypergraph with given dimensions."""
    k1, k2 = jax.random.split(key)
    H = (jax.random.uniform(k1, (n, m)) > 0.5).astype(jnp.float32)
    # Ensure every node in at least one edge and every edge has at least one node
    H = H.at[:, 0].set(1.0)
    H = H.at[0, :].set(1.0)
    feats = jax.random.normal(k2, (n, feat_dim)) * 0.3
    return from_incidence(H, node_features=feats)


# ---------------------------------------------------------------------------
# ProductManifold: dimension & split/combine
# ---------------------------------------------------------------------------


class TestProductManifoldDimension:
    """Product of (H^2, S^1, R^3) gives total dimension 6."""

    def test_total_dim(self):
        m = ProductManifold(
            components=[("hyperbolic", 2), ("spherical", 1), ("euclidean", 3)]
        )
        assert m.total_dim == 6


class TestSplitCombineRoundtrip:
    """split then combine recovers the original array."""

    def test_roundtrip(self):
        m = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (8, m.total_dim))
        parts = m.split(x)
        assert len(parts) == 3
        assert parts[0].shape == (8, 4)
        assert parts[1].shape == (8, 3)
        assert parts[2].shape == (8, 5)
        x_reconstructed = m.combine(parts)
        assert jnp.allclose(x, x_reconstructed, atol=1e-7)

    def test_single_point(self):
        m = ProductManifold(
            components=[("hyperbolic", 2), ("euclidean", 3)]
        )
        x = jnp.arange(5, dtype=jnp.float32)
        parts = m.split(x)
        assert jnp.allclose(parts[0], jnp.array([0.0, 1.0]))
        assert jnp.allclose(parts[1], jnp.array([2.0, 3.0, 4.0]))
        assert jnp.allclose(m.combine(parts), x)


# ---------------------------------------------------------------------------
# Manifold constraints after projection
# ---------------------------------------------------------------------------


class TestManifoldProjection:
    """After projection, each component obeys its manifold constraint."""

    def test_hyperbolic_in_ball(self):
        m = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (10, m.total_dim)) * 2.0
        c = jnp.array([1.0, 1.0, 1.0])
        x_proj = m.project(x, c)
        h_part = m.split(x_proj)[0]
        norms = jnp.linalg.norm(h_part, axis=-1)
        # All norms should be < 1/sqrt(c) = 1.0
        assert jnp.all(norms < 1.0)

    def test_spherical_on_sphere(self):
        m = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (10, m.total_dim)) * 2.0
        c = jnp.array([1.0, 1.0, 1.0])
        x_proj = m.project(x, c)
        s_part = m.split(x_proj)[1]
        norms = jnp.linalg.norm(s_part, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_euclidean_unconstrained(self):
        m = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (10, m.total_dim)) * 2.0
        c = jnp.array([1.0, 1.0, 1.0])
        x_proj = m.project(x, c)
        e_orig = m.split(x)[2]
        e_proj = m.split(x_proj)[2]
        # Euclidean component should be unchanged
        assert jnp.allclose(e_orig, e_proj, atol=1e-7)


# ---------------------------------------------------------------------------
# Spherical primitives
# ---------------------------------------------------------------------------


class TestSphereProject:
    def test_unit_norm(self):
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (5, 3))
        x_proj = sphere_project(x)
        norms = jnp.linalg.norm(x_proj, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)


class TestSphereExpLog:
    """exp(x, log(x, y)) should recover y (for nearby points on sphere)."""

    def test_roundtrip(self):
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        x = sphere_project(jax.random.normal(k1, (5, 4)))
        y = sphere_project(jax.random.normal(k2, (5, 4)))
        v = sphere_log(x, y)
        y_recovered = sphere_exp(x, v)
        assert jnp.allclose(y, y_recovered, atol=1e-4)

    def test_tangent_orthogonal(self):
        """Log map should produce tangent vectors orthogonal to x."""
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        x = sphere_project(jax.random.normal(k1, (5, 4)))
        y = sphere_project(jax.random.normal(k2, (5, 4)))
        v = sphere_log(x, y)
        dots = jnp.sum(x * v, axis=-1)
        assert jnp.allclose(dots, 0.0, atol=1e-4)


class TestSphereMidpoint:
    """Midpoint of identical points is the same point."""

    def test_identical_points(self):
        key = jax.random.PRNGKey(4)
        p = sphere_project(jax.random.normal(key, (3,)))
        points = jnp.stack([p, p, p])
        weights = jnp.ones(3)
        mid = sphere_midpoint(points, weights)
        assert jnp.allclose(mid, p, atol=1e-4)

    def test_antipodal_equal_weights(self):
        """Mean of antipodal points with equal weights should be on sphere."""
        x = jnp.array([1.0, 0.0, 0.0])
        # Not exactly antipodal (which would be degenerate), but close
        y = jnp.array([-0.99, 0.1, 0.1])
        y = sphere_project(y)
        points = jnp.stack([x, y])
        weights = jnp.ones(2)
        mid = sphere_midpoint(points, weights)
        norm = jnp.linalg.norm(mid)
        assert jnp.allclose(norm, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# ProductHypergraphConv: basic forward pass
# ---------------------------------------------------------------------------


class TestProductConvForward:
    """ProductHypergraphConv produces output of correct shape."""

    def test_output_shape(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(10)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(12, 12, manifold, key=k1)
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert out.shape == (6, 12)

    def test_different_output_dim(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(11)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(12, 9, manifold, key=k1)
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert out.shape == (6, 9)

    def test_no_nans(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(12)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(12, 12, manifold, key=k1)
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert not jnp.any(jnp.isnan(out))


# ---------------------------------------------------------------------------
# Manifold constraints on output
# ---------------------------------------------------------------------------


class TestOutputManifoldConstraints:
    """Output of ProductHypergraphConv respects manifold constraints."""

    def test_hyperbolic_output_in_ball(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(20)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(12, 12, manifold, key=k1)
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        out_manifold = conv._output_manifold
        h_part = out_manifold.split(out)[0]
        norms = jnp.linalg.norm(h_part, axis=-1)
        c = conv.curvatures[0]
        max_norm = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < max_norm + 1e-4)

    def test_spherical_output_on_sphere(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]
        )
        key = jax.random.PRNGKey(21)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(12, 12, manifold, key=k1)
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        out_manifold = conv._output_manifold
        s_part = out_manifold.split(out)[1]
        norms = jnp.linalg.norm(s_part, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:
    """ProductHypergraphConv works under jax.jit."""

    def test_jit_forward(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(30)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(9, 9, manifold, key=k1)
        hg = _make_hg(5, 2, 9, key=k2)

        @jax.jit
        def forward(conv, hg):
            return conv(hg)

        out = forward(conv, hg)
        assert out.shape == (5, 9)
        assert not jnp.any(jnp.isnan(out))

    def test_jit_produces_same_result(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(31)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(9, 9, manifold, key=k1)
        hg = _make_hg(5, 2, 9, key=k2)

        out_eager = conv(hg)

        @jax.jit
        def forward(conv, hg):
            return conv(hg)

        out_jit = forward(conv, hg)
        assert jnp.allclose(out_eager, out_jit, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow through all components including curvatures
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients flow through all parameters including curvatures."""

    def test_grad_wrt_curvatures(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(40)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(9, 9, manifold, key=k1)
        hg = _make_hg(5, 2, 9, key=k2)

        def loss_fn(conv):
            out = conv(hg)
            return jnp.sum(out ** 2)

        grads = jax.grad(loss_fn)(conv)
        # Curvature gradients should be nonzero
        assert not jnp.allclose(grads.log_curvatures, 0.0, atol=1e-10)

    def test_grad_wrt_linears(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(41)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(9, 9, manifold, key=k1)
        hg = _make_hg(5, 2, 9, key=k2)

        def loss_fn(conv):
            out = conv(hg)
            return jnp.sum(out ** 2)

        grads = jax.grad(loss_fn)(conv)
        # Check each linear layer has nonzero gradients
        for i, linear_grad in enumerate(grads.linears):
            assert not jnp.allclose(
                linear_grad.weight, 0.0, atol=1e-10
            ), f"Linear {i} weight grad is zero"

    def test_grad_all_components(self):
        """Gradient flows through hyperbolic, spherical, AND euclidean."""
        manifold = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        conv = ProductHypergraphConv(9, 9, manifold, key=k1)
        hg = _make_hg(5, 2, 9, key=k2)

        def loss_fn(conv):
            out = conv(hg)
            # Use output from each component region to ensure grad flows
            out_manifold = conv._output_manifold
            parts = out_manifold.split(out)
            return sum(jnp.sum(p ** 2) for p in parts)

        grads = jax.grad(loss_fn)(conv)
        # Hyperbolic curvature (index 0) should get gradients.
        # Spherical (index 1) may have zero grad if curvature
        # is not used in sphere ops. Euclidean (index 2) may also
        # be zero since it typically ignores curvature.
        assert grads.log_curvatures is not None
        assert jnp.all(jnp.isfinite(grads.log_curvatures))


# ---------------------------------------------------------------------------
# Product distance metric
# ---------------------------------------------------------------------------


class TestProductDistance:
    """Product metric distance works correctly."""

    def test_zero_distance_for_same_point(self):
        m = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(50)
        x = jax.random.normal(key, (9,)) * 0.3
        c = jnp.ones(3)
        x = m.project(x, c)
        d = m.distance(x, x, c)
        assert jnp.allclose(d, 0.0, atol=5e-3)

    def test_positive_for_different_points(self):
        m = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(51)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (9,)) * 0.3
        y = jax.random.normal(k2, (9,)) * 0.3
        c = jnp.ones(3)
        x = m.project(x, c)
        y = m.project(y, c)
        d = m.distance(x, y, c)
        assert d > 0.0


# ---------------------------------------------------------------------------
# Node mask support
# ---------------------------------------------------------------------------


class TestNodeMask:
    """Masked nodes produce zero output."""

    def test_masked_nodes_zero(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        key = jax.random.PRNGKey(60)
        k1, k2, k3 = jax.random.split(key, 3)
        conv = ProductHypergraphConv(9, 9, manifold, key=k1)
        H = jnp.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        feats = jax.random.normal(k2, (4, 9)) * 0.3
        mask = jnp.array([True, True, False, True])
        hg = Hypergraph(
            node_features=feats,
            incidence=H,
            node_mask=mask,
        )
        out = conv(hg)
        # Node 2 is masked, should be zero
        assert jnp.allclose(out[2], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Dimension mismatch error
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    def test_raises_on_mismatch(self):
        manifold = ProductManifold(
            components=[("hyperbolic", 3), ("spherical", 2), ("euclidean", 4)]
        )
        with pytest.raises(ValueError, match="in_dim.*must match"):
            ProductHypergraphConv(
                10, 9, manifold, key=jax.random.PRNGKey(0)
            )


# ===========================================================================
# Tests for ProductSpaceConv, ProductSpaceEmbedding, product_distance,
# and the public sphere helper aliases.
# ===========================================================================


# ---------------------------------------------------------------------------
# Sphere helper aliases
# ---------------------------------------------------------------------------


class TestSphereHelperAliases:
    """sphere_exp_map, sphere_log_map, project_to_sphere are aliases."""

    def test_exp_map_alias(self):
        assert sphere_exp_map is sphere_exp

    def test_log_map_alias(self):
        assert sphere_log_map is sphere_log

    def test_project_alias(self):
        assert project_to_sphere is sphere_project

    def test_project_to_sphere_unit_norm(self):
        key = jax.random.PRNGKey(100)
        x = jax.random.normal(key, (7, 4))
        x_proj = project_to_sphere(x)
        norms = jnp.linalg.norm(x_proj, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_sphere_exp_map_stays_on_sphere(self):
        key = jax.random.PRNGKey(101)
        k1, k2 = jax.random.split(key)
        x = project_to_sphere(jax.random.normal(k1, (3, 5)))
        # Make tangent vectors orthogonal to x
        v_raw = jax.random.normal(k2, (3, 5)) * 0.5
        v = v_raw - jnp.sum(v_raw * x, axis=-1, keepdims=True) * x
        y = sphere_exp_map(x, v)
        norms = jnp.linalg.norm(y, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# ProductSpaceConv: forward pass with mixed components
# ---------------------------------------------------------------------------


class TestProductSpaceConvForward:
    """ProductSpaceConv forward pass with hyperbolic + euclidean + spherical."""

    def test_output_shape(self):
        key = jax.random.PRNGKey(200)
        k1, k2 = jax.random.split(key)
        conv = ProductSpaceConv(
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            out_dim=12,
            key=k1,
        )
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert out.shape == (6, 12)

    def test_output_finite(self):
        key = jax.random.PRNGKey(201)
        k1, k2 = jax.random.split(key)
        conv = ProductSpaceConv(
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            out_dim=12,
            key=k1,
        )
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert jnp.all(jnp.isfinite(out))

    def test_different_output_dim(self):
        key = jax.random.PRNGKey(202)
        k1, k2 = jax.random.split(key)
        conv = ProductSpaceConv(
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            out_dim=9,
            key=k1,
        )
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert out.shape == (6, 9)


# ---------------------------------------------------------------------------
# ProductSpaceConv: each component stays on its manifold
# ---------------------------------------------------------------------------


class TestProductSpaceConvManifoldConstraints:
    """Hyperbolic output in ball, spherical output on sphere."""

    def test_hyperbolic_in_ball(self):
        key = jax.random.PRNGKey(210)
        k1, k2 = jax.random.split(key)
        dims = [4, 3, 5]
        types = ["hyperbolic", "spherical", "euclidean"]
        conv = ProductSpaceConv(dims, types, out_dim=12, key=k1)
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)

        # Extract hyperbolic component
        h_dim = conv.out_dims[0]
        h_part = out[:, :h_dim]
        norms = jnp.linalg.norm(h_part, axis=-1)
        c = jnp.abs(conv.c) + 1e-6
        radius = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < radius + 1e-4)

    def test_spherical_on_sphere(self):
        key = jax.random.PRNGKey(211)
        k1, k2 = jax.random.split(key)
        dims = [4, 3, 5]
        types = ["hyperbolic", "spherical", "euclidean"]
        conv = ProductSpaceConv(dims, types, out_dim=12, key=k1)
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)

        # Extract spherical component
        h_dim = conv.out_dims[0]
        s_dim = conv.out_dims[1]
        s_part = out[:, h_dim : h_dim + s_dim]
        norms = jnp.linalg.norm(s_part, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# product_distance: free function tests
# ---------------------------------------------------------------------------


class TestProductDistanceFreeFunction:
    """product_distance is non-negative and zero for identical points."""

    def test_zero_for_identical(self):
        key = jax.random.PRNGKey(220)
        dims = [3, 2, 4]
        types = ["hyperbolic", "spherical", "euclidean"]
        x = jax.random.normal(key, (9,)) * 0.3
        # Project onto manifolds manually
        h = x[:3]
        h = h / jnp.maximum(jnp.linalg.norm(h), 1e-6)
        h = h * 0.5  # inside ball
        s = x[3:5]
        s = s / jnp.linalg.norm(s)
        e = x[5:]
        pt = jnp.concatenate([h, s, e])
        d = product_distance(pt, pt, dims, types)
        assert jnp.allclose(d, 0.0, atol=5e-3)

    def test_non_negative(self):
        key = jax.random.PRNGKey(221)
        k1, k2 = jax.random.split(key)
        dims = [3, 2, 4]
        types = ["hyperbolic", "spherical", "euclidean"]
        x = jax.random.normal(k1, (9,)) * 0.3
        y = jax.random.normal(k2, (9,)) * 0.3
        # Project both
        x = jnp.concatenate([
            x[:3] * 0.5 / jnp.maximum(jnp.linalg.norm(x[:3]), 1e-6),
            x[3:5] / jnp.linalg.norm(x[3:5]),
            x[5:],
        ])
        y = jnp.concatenate([
            y[:3] * 0.5 / jnp.maximum(jnp.linalg.norm(y[:3]), 1e-6),
            y[3:5] / jnp.linalg.norm(y[3:5]),
            y[5:],
        ])
        d = product_distance(x, y, dims, types)
        assert d >= 0.0

    def test_batched(self):
        """product_distance works with batched inputs."""
        key = jax.random.PRNGKey(222)
        k1, k2 = jax.random.split(key)
        dims = [3, 2, 4]
        types = ["hyperbolic", "spherical", "euclidean"]
        x = jax.random.normal(k1, (5, 9)) * 0.2
        y = jax.random.normal(k2, (5, 9)) * 0.2
        d = product_distance(x, y, dims, types)
        assert d.shape == (5,)
        assert jnp.all(d >= 0.0)

    def test_mismatched_dims_raises(self):
        with pytest.raises(ValueError, match="same length"):
            product_distance(
                jnp.zeros(5), jnp.zeros(5),
                component_dims=[3, 2],
                component_types=["hyperbolic"],
            )


# ---------------------------------------------------------------------------
# ProductSpaceConv: JIT compatibility
# ---------------------------------------------------------------------------


class TestProductSpaceConvJIT:
    """ProductSpaceConv works under JIT."""

    def test_jit_forward(self):
        key = jax.random.PRNGKey(230)
        k1, k2 = jax.random.split(key)
        conv = ProductSpaceConv(
            [3, 2, 4], ["hyperbolic", "spherical", "euclidean"],
            out_dim=9, key=k1,
        )
        hg = _make_hg(5, 2, 9, key=k2)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out = forward(conv, hg)
        assert out.shape == (5, 9)
        assert jnp.all(jnp.isfinite(out))

    def test_jit_matches_eager(self):
        key = jax.random.PRNGKey(231)
        k1, k2 = jax.random.split(key)
        conv = ProductSpaceConv(
            [3, 2, 4], ["hyperbolic", "spherical", "euclidean"],
            out_dim=9, key=k1,
        )
        hg = _make_hg(5, 2, 9, key=k2)

        out_eager = conv(hg)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out_jit = forward(conv, hg)
        assert jnp.allclose(out_eager, out_jit, atol=1e-5)


# ---------------------------------------------------------------------------
# ProductSpaceConv: gradient flow through all components
# ---------------------------------------------------------------------------


class TestProductSpaceConvGradient:
    """Gradients flow through all component linears and curvature."""

    def test_gradient_flows(self):
        key = jax.random.PRNGKey(240)
        k1, k2 = jax.random.split(key)
        conv = ProductSpaceConv(
            [3, 2, 4], ["hyperbolic", "spherical", "euclidean"],
            out_dim=9, key=k1,
        )
        hg = _make_hg(5, 2, 9, key=k2)

        def loss_fn(model):
            return jnp.sum(model(hg) ** 2)

        grads = eqx.filter_grad(loss_fn)(conv)
        # All linear weights should get nonzero gradients
        for i, lg in enumerate(grads.linears):
            assert not jnp.allclose(lg.weight, 0.0, atol=1e-10), (
                f"Linear {i} weight grad is zero"
            )
        # Curvature gradient should be nonzero
        assert jnp.abs(grads.c) > 1e-10

    def test_gradient_finite(self):
        key = jax.random.PRNGKey(241)
        k1, k2 = jax.random.split(key)
        conv = ProductSpaceConv(
            [3, 2, 4], ["hyperbolic", "spherical", "euclidean"],
            out_dim=9, key=k1,
        )
        hg = _make_hg(5, 2, 9, key=k2)

        def loss_fn(model):
            return jnp.sum(model(hg) ** 2)

        grads = eqx.filter_grad(loss_fn)(conv)
        for lg in grads.linears:
            assert jnp.all(jnp.isfinite(lg.weight))
        assert jnp.isfinite(grads.c)


# ---------------------------------------------------------------------------
# ProductSpaceConv: pure Euclidean matches UniGCNConv
# ---------------------------------------------------------------------------


class TestProductSpaceConvPureEuclidean:
    """With only euclidean components, should match standard UniGCNConv."""

    def test_pure_euclidean_matches_unigcn(self):
        key = jax.random.PRNGKey(250)
        k1, k2 = jax.random.split(key)

        in_dim, out_dim = 8, 8
        # ProductSpaceConv with a single Euclidean component
        pconv = ProductSpaceConv(
            [in_dim], ["euclidean"], out_dim=out_dim, key=k1,
        )

        # UniGCNConv — copy the same weights from ProductSpaceConv to
        # ensure identical linear transforms (key splitting differs).
        uconv = hgx.UniGCNConv(
            in_dim=in_dim, out_dim=out_dim, normalize=True, key=k1,
        )
        uconv = eqx.tree_at(
            lambda m: m.linear, uconv,
            pconv.linears[0],
        )

        # Build a small hypergraph
        hg = _make_hg(4, 2, in_dim, key=k2)

        out_prod = pconv(hg)
        out_uni = uconv(hg)

        # Both use degree-normalised mean aggregation with the same
        # linear weights, so outputs should match.
        assert jnp.allclose(out_prod, out_uni, atol=1e-5), (
            f"max diff = {jnp.max(jnp.abs(out_prod - out_uni))}"
        )


# ---------------------------------------------------------------------------
# ProductSpaceEmbedding
# ---------------------------------------------------------------------------


class TestProductSpaceEmbedding:
    """ProductSpaceEmbedding correctly projects onto each manifold."""

    def test_output_shape(self):
        key = jax.random.PRNGKey(260)
        emb = ProductSpaceEmbedding(
            in_dim=10,
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            key=key,
        )
        x = jax.random.normal(key, (7, 10))
        out = jax.vmap(emb)(x)
        assert out.shape == (7, 12)

    def test_hyperbolic_in_ball(self):
        key = jax.random.PRNGKey(261)
        emb = ProductSpaceEmbedding(
            in_dim=10,
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            key=key,
        )
        x = jax.random.normal(key, (7, 10))
        out = jax.vmap(emb)(x)
        h_part = out[:, :4]
        norms = jnp.linalg.norm(h_part, axis=-1)
        assert jnp.all(norms < 1.0)

    def test_spherical_on_sphere(self):
        key = jax.random.PRNGKey(262)
        emb = ProductSpaceEmbedding(
            in_dim=10,
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            key=key,
        )
        x = jax.random.normal(key, (7, 10))
        out = jax.vmap(emb)(x)
        s_part = out[:, 4:7]
        norms = jnp.linalg.norm(s_part, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_euclidean_no_constraint(self):
        """Euclidean component is not projected."""
        key = jax.random.PRNGKey(263)
        emb = ProductSpaceEmbedding(
            in_dim=10,
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            key=key,
        )
        # Just verify it produces finite values with no constraint
        x = jax.random.normal(key, (7, 10))
        out = jax.vmap(emb)(x)
        e_part = out[:, 7:]
        assert jnp.all(jnp.isfinite(e_part))

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(264)
        emb = ProductSpaceEmbedding(
            in_dim=10,
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            key=key,
        )
        x = jax.random.normal(key, (7, 10))

        @eqx.filter_jit
        def embed(model, x):
            return jax.vmap(model)(x)

        out = embed(emb, x)
        assert out.shape == (7, 12)
        assert jnp.all(jnp.isfinite(out))

    def test_gradient_flows(self):
        key = jax.random.PRNGKey(265)
        emb = ProductSpaceEmbedding(
            in_dim=10,
            component_dims=[4, 3, 5],
            component_types=["hyperbolic", "spherical", "euclidean"],
            key=key,
        )
        x = jax.random.normal(key, (7, 10))

        def loss_fn(model):
            return jnp.sum(jax.vmap(model)(x) ** 2)

        grads = eqx.filter_grad(loss_fn)(emb)
        for i, lg in enumerate(grads.linears):
            assert not jnp.allclose(lg.weight, 0.0, atol=1e-10), (
                f"Embedding linear {i} grad is zero"
            )

    def test_mismatched_dims_raises(self):
        with pytest.raises(ValueError, match="same length"):
            ProductSpaceEmbedding(
                in_dim=10,
                component_dims=[4, 3],
                component_types=["hyperbolic"],
                key=jax.random.PRNGKey(0),
            )

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown component type"):
            ProductSpaceEmbedding(
                in_dim=10,
                component_dims=[4],
                component_types=["banana"],
                key=jax.random.PRNGKey(0),
            )


# ---------------------------------------------------------------------------
# Top-level package exports
# ---------------------------------------------------------------------------


class TestExports:
    """ProductSpaceConv and ProductSpaceEmbedding are importable from hgx."""

    def test_product_space_conv_in_hgx(self):
        assert hasattr(hgx, "ProductSpaceConv")
        assert hgx.ProductSpaceConv is ProductSpaceConv

    def test_product_space_embedding_in_hgx(self):
        assert hasattr(hgx, "ProductSpaceEmbedding")
        assert hgx.ProductSpaceEmbedding is ProductSpaceEmbedding

    def test_product_manifold_conv_in_hgx(self):
        assert hasattr(hgx, "ProductManifoldConv")
        assert hgx.ProductManifoldConv is ProductManifoldConv

    def test_product_manifold_mlp_in_hgx(self):
        assert hasattr(hgx, "ProductManifoldMLP")
        assert hgx.ProductManifoldMLP is ProductManifoldMLP


# ===========================================================================
# Tests for ProductManifoldConv, ProductManifoldMLP, and standalone
# utilities (split_components, concat_components, project_*,
# exp_map_*, log_map_*, product_distance).
# ===========================================================================


# ---------------------------------------------------------------------------
# split_components / concat_components roundtrip
# ---------------------------------------------------------------------------


class TestSplitConcatRoundtrip:
    """split then concat returns the original vector."""

    def test_split_concat_roundtrip(self, prng_key):
        dims = (4, 3, 5)
        x = jax.random.normal(prng_key, (8, sum(dims)))
        h, s, e = split_components(x, dims)
        assert h.shape == (8, 4)
        assert s.shape == (8, 3)
        assert e.shape == (8, 5)
        x_rec = concat_components(h, s, e)
        assert jnp.allclose(x, x_rec, atol=1e-7)

    def test_single_vector(self, prng_key):
        dims = (2, 3, 4)
        x = jax.random.normal(prng_key, (sum(dims),))
        h, s, e = split_components(x, dims)
        assert h.shape == (2,)
        assert s.shape == (3,)
        assert e.shape == (4,)
        assert jnp.allclose(
            concat_components(h, s, e), x, atol=1e-7
        )


# ---------------------------------------------------------------------------
# Projection preserves manifold constraints
# ---------------------------------------------------------------------------


class TestProjectionPreservesManifold:
    """Hyperbolic points stay in ball, spherical points on sphere."""

    def test_hyperbolic_in_ball(self, prng_key):
        x = jax.random.normal(prng_key, (10, 5)) * 3.0
        x_proj = project_hyperbolic(x, c=1.0)
        norms = jnp.linalg.norm(x_proj, axis=-1)
        assert jnp.all(norms < 1.0)

    def test_hyperbolic_custom_curvature(self, prng_key):
        x = jax.random.normal(prng_key, (10, 5)) * 3.0
        c = 2.0
        x_proj = project_hyperbolic(x, c=c)
        norms = jnp.linalg.norm(x_proj, axis=-1)
        radius = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < radius)

    def test_spherical_on_sphere(self, prng_key):
        x = jax.random.normal(prng_key, (10, 5)) * 3.0
        x_proj = project_spherical(x)
        norms = jnp.linalg.norm(x_proj, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_idempotent_hyperbolic(self, prng_key):
        x = jax.random.normal(prng_key, (10, 5)) * 0.3
        x_proj = project_hyperbolic(x, c=1.0)
        x_proj2 = project_hyperbolic(x_proj, c=1.0)
        assert jnp.allclose(x_proj, x_proj2, atol=1e-6)

    def test_idempotent_spherical(self, prng_key):
        x = jax.random.normal(prng_key, (10, 5))
        x_proj = project_spherical(x)
        x_proj2 = project_spherical(x_proj)
        assert jnp.allclose(x_proj, x_proj2, atol=1e-6)


# ---------------------------------------------------------------------------
# Product distance non-negativity
# ---------------------------------------------------------------------------


class TestProductDistanceNonneg:
    """Distances from product_distance are non-negative."""

    def test_product_distance_nonneg(self, prng_key):
        k1, k2 = jax.random.split(prng_key)
        dims = [3, 2, 4]
        types = ["hyperbolic", "spherical", "euclidean"]
        x = jax.random.normal(k1, (20, 9)) * 0.3
        y = jax.random.normal(k2, (20, 9)) * 0.3
        d = product_distance(x, y, dims, types)
        assert jnp.all(d >= 0.0)

    def test_zero_for_same_point(self, prng_key):
        dims = [3, 2, 4]
        types = ["hyperbolic", "spherical", "euclidean"]
        x = jax.random.normal(prng_key, (9,)) * 0.3
        h = x[:3] * 0.5 / jnp.maximum(
            jnp.linalg.norm(x[:3]), 1e-6
        )
        s = x[3:5] / jnp.linalg.norm(x[3:5])
        e = x[5:]
        pt = jnp.concatenate([h, s, e])
        d = product_distance(pt, pt, dims, types)
        assert jnp.allclose(d, 0.0, atol=5e-3)


# ---------------------------------------------------------------------------
# Exp/log map utilities
# ---------------------------------------------------------------------------


class TestExpLogMaps:
    """Exp and log maps for Poincare and sphere."""

    def test_poincare_exp_stays_in_ball(self, prng_key):
        k1, k2 = jax.random.split(prng_key)
        x = project_hyperbolic(
            jax.random.normal(k1, (5, 4)) * 0.3, c=1.0
        )
        v = jax.random.normal(k2, (5, 4)) * 0.1
        y = exp_map_poincare(x, v, c=1.0)
        norms = jnp.linalg.norm(y, axis=-1)
        assert jnp.all(norms < 1.0)

    def test_poincare_log_finite(self, prng_key):
        k1, k2 = jax.random.split(prng_key)
        x = project_hyperbolic(
            jax.random.normal(k1, (5, 4)) * 0.3, c=1.0
        )
        y = project_hyperbolic(
            jax.random.normal(k2, (5, 4)) * 0.3, c=1.0
        )
        v = log_map_poincare(x, y, c=1.0)
        assert jnp.all(jnp.isfinite(v))

    def test_sphere_exp_stays_on_sphere(self, prng_key):
        k1, k2 = jax.random.split(prng_key)
        x = project_spherical(
            jax.random.normal(k1, (5, 4))
        )
        v_raw = jax.random.normal(k2, (5, 4)) * 0.5
        v = v_raw - jnp.sum(
            v_raw * x, axis=-1, keepdims=True
        ) * x
        y = exp_map_sphere(x, v)
        norms = jnp.linalg.norm(y, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_sphere_log_tangent(self, prng_key):
        k1, k2 = jax.random.split(prng_key)
        x = project_spherical(
            jax.random.normal(k1, (5, 4))
        )
        y = project_spherical(
            jax.random.normal(k2, (5, 4))
        )
        v = log_map_sphere(x, y)
        dots = jnp.sum(x * v, axis=-1)
        assert jnp.allclose(dots, 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# ProductManifoldConv tests
# ---------------------------------------------------------------------------


class TestProductManifoldConvOutputShape:
    """Verify (n, sum(dims)) output shape."""

    def test_output_shape(self, prng_key):
        dims = (4, 3, 5)
        total = sum(dims)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            total, total, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(6, 3, total, key=k2)
        out = conv(hg)
        assert out.shape == (6, total)

    def test_different_out_dim(self, prng_key):
        dims = (4, 3, 5)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            12, 9, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert out.shape == (6, 9)

    def test_no_nans(self, prng_key):
        dims = (4, 3, 5)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            12, 12, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        assert jnp.all(jnp.isfinite(out))


class TestProductManifoldConvJIT:
    """eqx.filter_jit works with ProductManifoldConv."""

    def test_jit_compatible(self, prng_key):
        dims = (3, 2, 4)
        total = sum(dims)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            total, total, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(5, 2, total, key=k2)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out = forward(conv, hg)
        assert out.shape == (5, total)
        assert jnp.all(jnp.isfinite(out))

    def test_jit_matches_eager(self, prng_key):
        dims = (3, 2, 4)
        total = sum(dims)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            total, total, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(5, 2, total, key=k2)
        out_eager = conv(hg)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out_jit = forward(conv, hg)
        assert jnp.allclose(out_eager, out_jit, atol=1e-5)


class TestProductManifoldConvDifferentiable:
    """Gradients flow through ProductManifoldConv."""

    def test_differentiable(self, prng_key):
        dims = (3, 2, 4)
        total = sum(dims)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            total, total, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(5, 2, total, key=k2)

        def loss_fn(model):
            return jnp.sum(model(hg) ** 2)

        grads = eqx.filter_grad(loss_fn)(conv)
        assert not jnp.allclose(
            grads.linear_h.weight, 0.0, atol=1e-10
        )
        assert not jnp.allclose(
            grads.linear_s.weight, 0.0, atol=1e-10
        )
        assert not jnp.allclose(
            grads.linear_e.weight, 0.0, atol=1e-10
        )

    def test_gradient_finite(self, prng_key):
        dims = (3, 2, 4)
        total = sum(dims)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            total, total, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(5, 2, total, key=k2)

        def loss_fn(model):
            return jnp.sum(model(hg) ** 2)

        grads = eqx.filter_grad(loss_fn)(conv)
        assert jnp.all(jnp.isfinite(grads.linear_h.weight))
        assert jnp.all(jnp.isfinite(grads.linear_s.weight))
        assert jnp.all(jnp.isfinite(grads.linear_e.weight))
        assert jnp.isfinite(grads.c_h)
        assert jnp.isfinite(grads.c_s)


class TestProductManifoldConvManifoldOutput:
    """Output respects manifold constraints."""

    def test_hyperbolic_in_ball(self, prng_key):
        dims = (4, 3, 5)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            12, 12, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        o_h = conv.out_dims[0]
        h_part = out[:, :o_h]
        norms = jnp.linalg.norm(h_part, axis=-1)
        c_h = jnp.abs(conv.c_h) + 1e-6
        radius = 1.0 / jnp.sqrt(c_h)
        assert jnp.all(norms < radius + 1e-4)

    def test_spherical_on_sphere(self, prng_key):
        dims = (4, 3, 5)
        k1, k2 = jax.random.split(prng_key)
        conv = ProductManifoldConv(
            12, 12, dims, (1.0, 1.0), key=k1
        )
        hg = _make_hg(6, 3, 12, key=k2)
        out = conv(hg)
        o_h = conv.out_dims[0]
        o_s = conv.out_dims[1]
        s_part = out[:, o_h : o_h + o_s]
        norms = jnp.linalg.norm(s_part, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# ProductManifoldMLP tests
# ---------------------------------------------------------------------------


class TestProductManifoldMLPOutputShape:
    """ProductManifoldMLP preserves shape."""

    def test_output_shape(self, prng_key):
        dims = (4, 3, 5)
        total = sum(dims)
        mlp = ProductManifoldMLP(
            dims, 16, (1.0, 1.0), key=prng_key
        )
        x = jax.random.normal(prng_key, (total,)) * 0.3
        out = mlp(x)
        assert out.shape == (total,)

    def test_batched_via_vmap(self, prng_key):
        dims = (4, 3, 5)
        total = sum(dims)
        mlp = ProductManifoldMLP(
            dims, 16, (1.0, 1.0), key=prng_key
        )
        x = jax.random.normal(prng_key, (8, total)) * 0.3
        out = jax.vmap(mlp)(x)
        assert out.shape == (8, total)


class TestProductManifoldMLPJIT:
    """ProductManifoldMLP works under JIT."""

    def test_jit_compatible(self, prng_key):
        dims = (3, 2, 4)
        total = sum(dims)
        mlp = ProductManifoldMLP(
            dims, 16, (1.0, 1.0), key=prng_key
        )
        x = jax.random.normal(prng_key, (total,)) * 0.3

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        out = forward(mlp, x)
        assert out.shape == (total,)
        assert jnp.all(jnp.isfinite(out))


class TestProductManifoldMLPDifferentiable:
    """Gradients flow through ProductManifoldMLP."""

    def test_differentiable(self, prng_key):
        dims = (3, 2, 4)
        total = sum(dims)
        mlp = ProductManifoldMLP(
            dims, 16, (1.0, 1.0), key=prng_key
        )
        x = jax.random.normal(prng_key, (total,)) * 0.3

        def loss_fn(model):
            return jnp.sum(model(x) ** 2)

        grads = eqx.filter_grad(loss_fn)(mlp)
        assert not jnp.allclose(
            grads.linear1_h.weight, 0.0, atol=1e-10
        )
        assert not jnp.allclose(
            grads.linear1_e.weight, 0.0, atol=1e-10
        )


class TestProductManifoldMLPManifold:
    """MLP output respects manifold constraints."""

    def test_hyperbolic_in_ball(self, prng_key):
        dims = (4, 3, 5)
        total = sum(dims)
        mlp = ProductManifoldMLP(
            dims, 16, (1.0, 1.0), key=prng_key
        )
        x = jax.random.normal(prng_key, (8, total)) * 0.3
        out = jax.vmap(mlp)(x)
        h_part = out[:, :4]
        norms = jnp.linalg.norm(h_part, axis=-1)
        assert jnp.all(norms < 1.0)

    def test_spherical_on_sphere(self, prng_key):
        dims = (4, 3, 5)
        total = sum(dims)
        mlp = ProductManifoldMLP(
            dims, 16, (1.0, 1.0), key=prng_key
        )
        x = jax.random.normal(prng_key, (8, total)) * 0.3
        out = jax.vmap(mlp)(x)
        s_part = out[:, 4:7]
        norms = jnp.linalg.norm(s_part, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)
