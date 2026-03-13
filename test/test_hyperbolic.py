"""Tests for hyperbolic hypergraph convolution (Poincaré ball)."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._conv._hyperbolic import (
    expmap0,
    gyromidpoint,
    logmap0,
    mobius_add,
    PoincareHypergraphConv,
    project,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def poincare_hg():
    """4-node hypergraph with 2 hyperedges, features inside the ball."""
    H = jnp.array(
        [[1, 0], [1, 1], [1, 1], [0, 1]], dtype=jnp.float32
    )
    return hgx.from_incidence(
        H,
        node_features=jnp.ones((4, 8)) * 0.1,
        geometry="poincare",
    )


@pytest.fixture
def random_poincare_hg(key):
    """Random features inside the ball for a more rigorous test."""
    H = jnp.array(
        [[1, 0], [1, 1], [1, 1], [0, 1]], dtype=jnp.float32
    )
    x = jax.random.normal(key, (4, 8)) * 0.3  # small norm, inside ball
    return hgx.from_incidence(H, node_features=x, geometry="poincare")


# ---------------------------------------------------------------------------
# Poincaré ball primitives
# ---------------------------------------------------------------------------


class TestPrimitives:
    """Test exp/log maps, Möbius addition, projection, gyromidpoint."""

    def test_expmap0_logmap0_roundtrip(self):
        """exp₀ ∘ log₀ should be the identity on the ball."""
        c = jnp.array(1.0)
        x = jnp.array([0.1, 0.2, -0.3])
        v = logmap0(x, c)
        x_rec = expmap0(v, c)
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_logmap0_expmap0_roundtrip(self):
        """log₀ ∘ exp₀ should be the identity on tangent vectors."""
        c = jnp.array(1.0)
        v = jnp.array([0.5, -0.3, 0.1])
        x = expmap0(v, c)
        v_rec = logmap0(x, c)
        assert jnp.allclose(v, v_rec, atol=1e-5)

    def test_expmap0_origin(self):
        """exp₀(0) should be approximately 0."""
        c = jnp.array(1.0)
        v = jnp.zeros(4)
        x = expmap0(v, c)
        assert jnp.allclose(x, 0.0, atol=1e-4)

    def test_expmap0_stays_in_ball(self):
        """exp₀ output should be inside (or on boundary of) the ball.

        For very large tangent vectors, tanh saturates so the norm
        can reach the boundary; ``project`` is used after ``expmap0``
        in the conv layer to guarantee strict containment.
        """
        c = jnp.array(2.0)
        v = jnp.array([10.0, -10.0, 5.0])  # large tangent vector
        x = expmap0(v, c)
        radius = 1.0 / jnp.sqrt(c)
        assert jnp.linalg.norm(x) <= radius + 1e-6

    def test_project_inside_unchanged(self):
        """Points already in the ball should be unchanged."""
        c = jnp.array(1.0)
        x = jnp.array([0.1, 0.2, -0.3])  # norm ≈ 0.37 < 1.0
        x_proj = project(x, c)
        assert jnp.allclose(x, x_proj, atol=1e-5)

    def test_project_outside_clamps(self):
        """Points outside the ball should be clamped."""
        c = jnp.array(1.0)
        x = jnp.array([5.0, 5.0, 5.0])
        x_proj = project(x, c)
        assert jnp.linalg.norm(x_proj) < 1.0 / jnp.sqrt(c)

    def test_project_different_curvatures(self):
        """Higher curvature → smaller ball → tighter clamp."""
        x = jnp.array([0.5, 0.5])
        p1 = project(x, jnp.array(1.0))   # radius 1.0
        p4 = project(x, jnp.array(4.0))   # radius 0.5
        assert jnp.linalg.norm(p4) < jnp.linalg.norm(p1)

    def test_mobius_add_identity(self):
        """x ⊕ 0 = x."""
        c = jnp.array(1.0)
        x = jnp.array([0.1, 0.2])
        result = mobius_add(x, jnp.zeros(2), c)
        assert jnp.allclose(result, x, atol=1e-5)

    def test_mobius_add_stays_in_ball(self):
        """Möbius addition should keep results inside the ball."""
        c = jnp.array(1.0)
        x = jnp.array([0.3, 0.4])
        y = jnp.array([-0.2, 0.3])
        result = mobius_add(x, y, c)
        assert jnp.linalg.norm(result) < 1.0 / jnp.sqrt(c)

    def test_gyromidpoint_single_point(self):
        """Gyromidpoint of one point should return that point."""
        c = jnp.array(1.0)
        x = jnp.array([[0.1, 0.2]])  # (1, 2)
        membership = jnp.array([[1.0]])  # (1, 1)
        result = gyromidpoint(x, membership, c)
        assert jnp.allclose(result[0], x[0], atol=1e-4)

    def test_gyromidpoint_origin_symmetry(self):
        """Midpoint of symmetric points should be near origin."""
        c = jnp.array(1.0)
        x = jnp.array([[0.3, 0.0], [-0.3, 0.0]])  # (2, 2)
        membership = jnp.array([[1.0], [1.0]])  # (2, 1)
        result = gyromidpoint(x, membership, c)
        assert jnp.linalg.norm(result[0]) < 0.05

    def test_gyromidpoint_stays_in_ball(self):
        """Gyromidpoint should stay inside the ball."""
        c = jnp.array(1.0)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (10, 5)) * 0.3
        x = project(x, c)
        membership = jnp.ones((10, 3))
        result = gyromidpoint(x, membership, c)
        norms = jnp.linalg.norm(result, axis=-1)
        assert jnp.all(norms < 1.0 / jnp.sqrt(c))

    def test_gyromidpoint_empty_group(self):
        """Group with no members should give approximately zero."""
        c = jnp.array(1.0)
        x = jnp.array([[0.3, 0.2]])  # (1, 2)
        # (1, 2): group 0 has member, group 1 empty
        membership = jnp.array([[1.0, 0.0]])
        result = gyromidpoint(x, membership, c)
        # Group 0 should get the point
        assert jnp.allclose(result[0], x[0], atol=1e-4)
        # Group 1 (empty) should be near origin
        assert jnp.linalg.norm(result[1]) < 1e-4


# ---------------------------------------------------------------------------
# PoincareHypergraphConv layer
# ---------------------------------------------------------------------------


class TestPoincareConvForward:
    """Test forward pass properties."""

    def test_output_stays_in_ball(self, poincare_hg, key):
        """All output features must have norm < 1/√c."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)
        c = jnp.abs(conv.c) + 1e-6
        out = conv(poincare_hg)
        norms = jnp.linalg.norm(out, axis=-1)
        radius = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < radius)

    def test_output_stays_in_ball_random(self, random_poincare_hg, key):
        """Ball containment should hold for random inputs too."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)
        c = jnp.abs(conv.c) + 1e-6
        out = conv(random_poincare_hg)
        norms = jnp.linalg.norm(out, axis=-1)
        radius = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < radius)

    def test_output_stays_in_ball_high_curvature(self, poincare_hg, key):
        """Ball containment at c=4 (radius=0.5)."""
        conv = PoincareHypergraphConv(
            in_dim=8, out_dim=8, c_init=4.0, key=key
        )
        c = jnp.abs(conv.c) + 1e-6
        out = conv(poincare_hg)
        norms = jnp.linalg.norm(out, axis=-1)
        radius = 1.0 / jnp.sqrt(c)
        assert jnp.all(norms < radius)

    def test_output_shape(self, poincare_hg, key):
        """Output shape should be (num_nodes, out_dim)."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=16, key=key)
        out = conv(poincare_hg)
        assert out.shape == (4, 16)

    def test_output_finite(self, poincare_hg, key):
        """All output values should be finite."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)
        out = conv(poincare_hg)
        assert jnp.all(jnp.isfinite(out))


class TestPoincareConvJIT:
    """Test JIT compatibility."""

    def test_jit_consistent(self, poincare_hg, key):
        """JIT and eager should produce identical results."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out_jit = forward(conv, poincare_hg)
        out_eager = conv(poincare_hg)
        assert jnp.allclose(out_jit, out_eager, atol=1e-5)

    def test_jit_stays_in_ball(self, poincare_hg, key):
        """Ball containment should hold under JIT too."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        c = jnp.abs(conv.c) + 1e-6
        out = forward(conv, poincare_hg)
        norms = jnp.linalg.norm(out, axis=-1)
        assert jnp.all(norms < 1.0 / jnp.sqrt(c))


class TestPoincareConvGradient:
    """Test gradient flow through the layer."""

    def test_gradient_linear_weights(self, poincare_hg, key):
        """Linear weight gradients should be non-zero."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)

        def loss_fn(model):
            return jnp.sum(model(poincare_hg))

        grads = eqx.filter_grad(loss_fn)(conv)
        assert not jnp.allclose(grads.linear.weight, 0.0)

    def test_gradient_curvature(self, poincare_hg, key):
        """Curvature c should receive a non-zero gradient."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)

        def loss_fn(model):
            return jnp.sum(model(poincare_hg))

        grads = eqx.filter_grad(loss_fn)(conv)
        assert jnp.abs(grads.c) > 1e-8

    def test_gradient_finite(self, poincare_hg, key):
        """All gradients should be finite."""
        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)

        def loss_fn(model):
            return jnp.sum(model(poincare_hg))

        grads = eqx.filter_grad(loss_fn)(conv)
        assert jnp.all(jnp.isfinite(grads.linear.weight))
        assert jnp.isfinite(grads.c)


class TestPoincareVsEuclidean:
    """Test that Poincaré outputs differ from Euclidean UniGCN."""

    def test_differs_from_euclidean(self, poincare_hg, key):
        """Poincaré and Euclidean convolutions should give different outputs."""
        poincare_conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)
        euclidean_conv = hgx.UniGCNConv(in_dim=8, out_dim=8, key=key)

        out_hyp = poincare_conv(poincare_hg)
        out_euc = euclidean_conv(poincare_hg)

        assert not jnp.allclose(out_hyp, out_euc, atol=1e-3)

    def test_curvature_affects_output(self, poincare_hg, key):
        """Different curvature values should give different outputs."""
        conv1 = PoincareHypergraphConv(in_dim=8, out_dim=8, c_init=1.0, key=key)
        conv2 = eqx.tree_at(lambda m: m.c, conv1, jnp.array(4.0))

        out1 = conv1(poincare_hg)
        out2 = conv2(poincare_hg)

        assert not jnp.allclose(out1, out2)


class TestPoincareConvMasked:
    """Test behaviour with masked (pre-allocated) hypergraphs."""

    def test_masked_nodes_zero(self, key):
        """Inactive slots should have zero output features."""
        hg = hgx.from_edge_list(
            [(0, 1, 2), (1, 2, 3)],
            node_features=jnp.ones((4, 8)) * 0.1,
        )
        hg = hgx.preallocate(hg, max_nodes=6, max_edges=3)
        hg_p = hgx.Hypergraph(
            node_features=hg.node_features,
            incidence=hg.incidence,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry="poincare",
        )

        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)
        out = conv(hg_p)

        assert jnp.allclose(out[4:], 0.0)

    def test_active_nodes_nonzero(self, key):
        """Active nodes should have non-zero output features."""
        hg = hgx.from_edge_list(
            [(0, 1, 2), (1, 2, 3)],
            node_features=jnp.ones((4, 8)) * 0.1,
        )
        hg = hgx.preallocate(hg, max_nodes=6, max_edges=3)
        hg_p = hgx.Hypergraph(
            node_features=hg.node_features,
            incidence=hg.incidence,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry="poincare",
        )

        conv = PoincareHypergraphConv(in_dim=8, out_dim=8, key=key)
        out = conv(hg_p)

        assert not jnp.allclose(out[:4], 0.0)


class TestExportAccessible:
    """Verify the layer is accessible from the top-level package."""

    def test_import_from_hgx(self):
        """PoincareHypergraphConv should be importable from hgx."""
        assert hasattr(hgx, "PoincareHypergraphConv")
        assert hgx.PoincareHypergraphConv is PoincareHypergraphConv
