"""Tests for cellular sheaf neural networks on hypergraphs."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._conv._sheaf import learn_restriction_maps, SheafDiffusion, SheafHypergraphConv


class TestSheafHypergraphConv:
    """Test SheafHypergraphConv layer."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        """Output should have shape (n, in_dim)."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        conv = SheafHypergraphConv(
            in_dim=2, edge_stalk_dim=4, num_incidences=nnz, key=prng_key,
        )
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 2)

    def test_output_shape_larger_stalk(self, prng_key):
        """Edge stalk dim can differ from vertex stalk dim."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jnp.ones((3, 5))
        hg = hgx.from_incidence(H, node_features=features)
        nnz = int(jnp.sum(H > 0))
        conv = SheafHypergraphConv(
            in_dim=5, edge_stalk_dim=8, num_incidences=nnz, key=prng_key,
        )
        out = conv(hg)
        assert out.shape == (3, 5)

    def test_output_shape_smaller_stalk(self, prng_key):
        """Edge stalk dim smaller than vertex stalk dim."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        features = jnp.ones((3, 8))
        hg = hgx.from_incidence(H, node_features=features)
        nnz = int(jnp.sum(H > 0))
        conv = SheafHypergraphConv(
            in_dim=8, edge_stalk_dim=3, num_incidences=nnz, key=prng_key,
        )
        out = conv(hg)
        assert out.shape == (3, 8)

    def test_identity_maps_standard_diffusion(self, prng_key):
        """With identity restriction maps, sheaf diffusion should reduce to
        standard hypergraph Laplacian diffusion."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        d = 3
        features = jax.random.normal(prng_key, (4, d))
        hg = hgx.from_incidence(H, node_features=features)

        nnz = int(jnp.sum(H > 0))
        k1, k2 = jax.random.split(prng_key)
        conv = SheafHypergraphConv(
            in_dim=d, edge_stalk_dim=d, num_incidences=nnz, key=k1,
        )

        # Set restriction maps to identity
        identity_maps = jnp.broadcast_to(
            jnp.eye(d)[None, :, :], (nnz, d, d),
        )
        conv = eqx.tree_at(lambda c: c.restriction_maps, conv, identity_maps)
        # Zero out bias for clean comparison
        conv = eqx.tree_at(lambda c: c.bias, conv, jnp.zeros(d))
        # step_size = 1.0 (already default)

        out = conv(hg)

        # Manually compute standard Laplacian diffusion with mean aggregation:
        # For each (i,e): z_{i,e} = x_i (identity map)
        # z_e = mean_{j in e}(x_j)
        # Delta_i = sum_{e ni i} (x_i - z_e)
        # x_i' = x_i - Delta_i
        v_idx, e_idx = jnp.nonzero(H, size=nnz)
        x_inc = features[v_idx]
        z_e_sum = jnp.zeros((2, d)).at[e_idx].add(x_inc)
        e_deg = jnp.array([3.0, 3.0])
        z_e = z_e_sum / e_deg[:, None]
        diff = x_inc - z_e[e_idx]
        delta = jnp.zeros((4, d)).at[v_idx].add(diff)
        expected = features - delta

        assert jnp.allclose(out, expected, atol=1e-5)

    def test_jit_compatible(self, tiny_hypergraph, prng_key):
        """SheafHypergraphConv should work under eqx.filter_jit."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        conv = SheafHypergraphConv(
            in_dim=2, edge_stalk_dim=4, num_incidences=nnz, key=prng_key,
        )
        jit_conv = eqx.filter_jit(conv)
        out_jit = jit_conv(tiny_hypergraph)
        out_eager = conv(tiny_hypergraph)
        assert out_jit.shape == (4, 2)
        assert jnp.allclose(out_jit, out_eager, atol=1e-6)

    def test_gradient_flow(self, tiny_hypergraph, prng_key):
        """Gradients should flow through restriction maps, step_size, and bias."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        conv = SheafHypergraphConv(
            in_dim=2, edge_stalk_dim=4, num_incidences=nnz, key=prng_key,
        )

        @eqx.filter_grad
        def grad_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = grad_fn(conv)
        # Restriction maps should have non-zero gradients
        assert not jnp.allclose(grads.restriction_maps, 0.0)
        # Step size should have a gradient
        assert grads.step_size != 0.0
        # Bias should have a gradient
        assert not jnp.allclose(grads.bias, 0.0)

    def test_masked_node_gets_zero(self, prng_key):
        """Masked nodes should produce zero output."""
        H = jnp.ones((3, 1))
        mask = jnp.array([True, True, False])
        hg = hgx.Hypergraph(
            node_features=jnp.ones((3, 2)),
            incidence=H,
            node_mask=mask,
        )
        # After masking, only 2 nonzeros remain (nodes 0 and 1)
        H_masked = hg._masked_incidence()
        nnz = int(jnp.sum(H_masked > 0))
        conv = SheafHypergraphConv(
            in_dim=2, edge_stalk_dim=2, num_incidences=nnz, key=prng_key,
        )
        out = conv(hg)
        assert jnp.allclose(out[2], jnp.zeros(2))


class TestSheafDiffusion:
    """Test SheafDiffusion multi-step module."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        """Output should preserve input shape."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        diffusion = SheafDiffusion(
            num_steps=3,
            in_dim=2,
            edge_stalk_dim=4,
            num_incidences=nnz,
            share_maps=True,
            key=prng_key,
        )
        out = diffusion(tiny_hypergraph)
        assert out.shape == (4, 2)

    def test_shared_maps(self, tiny_hypergraph, prng_key):
        """With share_maps=True, all layers should share the same maps."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        diffusion = SheafDiffusion(
            num_steps=3,
            in_dim=2,
            edge_stalk_dim=4,
            num_incidences=nnz,
            share_maps=True,
            key=prng_key,
        )
        # All layers should reference the same restriction maps
        for i in range(1, 3):
            assert jnp.allclose(
                diffusion.layers[0].restriction_maps,
                diffusion.layers[i].restriction_maps,
            )

    def test_unshared_maps(self, tiny_hypergraph, prng_key):
        """With share_maps=False, layers should have different maps."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        diffusion = SheafDiffusion(
            num_steps=3,
            in_dim=2,
            edge_stalk_dim=4,
            num_incidences=nnz,
            share_maps=False,
            key=prng_key,
        )
        # At least some layers should have different maps
        assert not jnp.allclose(
            diffusion.layers[0].restriction_maps,
            diffusion.layers[1].restriction_maps,
        )

    def test_convergence_smoothing(self, prng_key):
        """Features should smooth under repeated sheaf diffusion with
        identity restriction maps (standard Laplacian diffusion)."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        d = 2
        # Start with heterogeneous features
        features = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H, node_features=features)

        nnz = int(jnp.sum(H > 0))
        diffusion = SheafDiffusion(
            num_steps=50,
            in_dim=d,
            edge_stalk_dim=d,
            num_incidences=nnz,
            share_maps=True,
            key=prng_key,
        )
        # Set all restriction maps to identity
        identity_maps = jnp.broadcast_to(
            jnp.eye(d)[None, :, :], (nnz, d, d),
        )
        # Set step_size small enough for stability
        step_size = jnp.array(0.3)
        zero_bias = jnp.zeros(d)

        # Update all layers (shared, so just update the one underlying layer)
        base_layer = diffusion.layers[0]
        base_layer = eqx.tree_at(
            lambda c: c.restriction_maps, base_layer, identity_maps,
        )
        base_layer = eqx.tree_at(lambda c: c.step_size, base_layer, step_size)
        base_layer = eqx.tree_at(lambda c: c.bias, base_layer, zero_bias)
        diffusion = eqx.tree_at(
            lambda d: d.layers,
            diffusion,
            tuple(base_layer for _ in range(50)),
        )

        out = diffusion(hg)

        # After many diffusion steps, features should be closer to each other.
        # Measure variance across nodes; it should decrease.
        initial_var = jnp.var(features)
        final_var = jnp.var(out)
        assert final_var < initial_var

    def test_jit_compatible(self, tiny_hypergraph, prng_key):
        """SheafDiffusion should work under eqx.filter_jit."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        diffusion = SheafDiffusion(
            num_steps=2,
            in_dim=2,
            edge_stalk_dim=3,
            num_incidences=nnz,
            share_maps=False,
            key=prng_key,
        )
        jit_diffusion = eqx.filter_jit(diffusion)
        out_jit = jit_diffusion(tiny_hypergraph)
        out_eager = diffusion(tiny_hypergraph)
        assert out_jit.shape == (4, 2)
        assert jnp.allclose(out_jit, out_eager, atol=1e-6)

    def test_gradient_flow(self, tiny_hypergraph, prng_key):
        """Gradients should flow through all diffusion steps."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        diffusion = SheafDiffusion(
            num_steps=2,
            in_dim=2,
            edge_stalk_dim=4,
            num_incidences=nnz,
            share_maps=False,
            key=prng_key,
        )

        @eqx.filter_grad
        def grad_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = grad_fn(diffusion)
        # Both layers should have non-zero gradients on restriction maps
        for i in range(2):
            assert not jnp.allclose(grads.layers[i].restriction_maps, 0.0)


class TestLearnRestrictionMaps:
    """Test the learn_restriction_maps initialization helper."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        """Should produce maps of shape (nnz, d_stalk, d_edge)."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        maps = learn_restriction_maps(tiny_hypergraph, target_dim=4, key=prng_key)
        assert maps.shape == (nnz, 2, 4)

    def test_output_shape_target_smaller(self, prng_key):
        """Target dim smaller than feature dim."""
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        features = jnp.ones((3, 5))
        hg = hgx.from_incidence(H, node_features=features)
        nnz = int(jnp.sum(H > 0))
        maps = learn_restriction_maps(hg, target_dim=2, key=prng_key)
        assert maps.shape == (nnz, 5, 2)

    def test_maps_usable_in_conv(self, tiny_hypergraph, prng_key):
        """Learned maps should plug into SheafHypergraphConv without error."""
        H = tiny_hypergraph.incidence
        nnz = int(jnp.sum(H > 0))
        k1, k2 = jax.random.split(prng_key)

        maps = learn_restriction_maps(tiny_hypergraph, target_dim=4, key=k1)
        conv = SheafHypergraphConv(
            in_dim=2, edge_stalk_dim=4, num_incidences=nnz, key=k2,
        )
        conv = eqx.tree_at(lambda c: c.restriction_maps, conv, maps)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 2)
        assert jnp.all(jnp.isfinite(out))


class TestDifferentStalkDimensions:
    """Test sheaf conv with various d_stalk != d_edge configurations."""

    @pytest.mark.parametrize("d_stalk,d_edge", [
        (2, 4),
        (4, 2),
        (3, 3),
        (1, 8),
        (8, 1),
    ])
    def test_various_dimensions(self, d_stalk, d_edge, prng_key):
        """Forward pass should work for any combination of stalk dims."""
        H = jnp.array([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        features = jax.random.normal(prng_key, (4, d_stalk))
        hg = hgx.from_incidence(H, node_features=features)
        nnz = int(jnp.sum(H > 0))

        k1, _ = jax.random.split(prng_key)
        conv = SheafHypergraphConv(
            in_dim=d_stalk, edge_stalk_dim=d_edge, num_incidences=nnz, key=k1,
        )
        out = conv(hg)
        assert out.shape == (4, d_stalk)
        assert jnp.all(jnp.isfinite(out))
