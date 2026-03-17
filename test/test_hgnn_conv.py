"""Tests for HGNNConv (project-after-aggregate) and HGNNSparseConv."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp


class TestHGNNConv:
    """Test HGNN convolution layer (project after aggregation)."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        conv = hgx.HGNNConv(in_dim=2, out_dim=8, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_output_shape_no_bias(self, tiny_hypergraph, prng_key):
        conv = hgx.HGNNConv(in_dim=2, out_dim=8, use_bias=False, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_unnormalized(self, tiny_hypergraph, prng_key):
        conv = hgx.HGNNConv(in_dim=2, out_dim=4, normalize=False, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 4)
        assert jnp.any(out != 0)

    def test_isolated_node_gets_zero(self, prng_key):
        """A node not in any hyperedge should get zero output."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],  # node 3: isolated
        ])
        features = jnp.ones((4, 2))
        hg = hgx.from_incidence(H, node_features=features)
        conv = hgx.HGNNConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg)
        assert jnp.allclose(out[3], jnp.zeros(4))

    def test_masked_node_gets_zero(self, prng_key):
        """Masked nodes should produce zero output."""
        H = jnp.ones((3, 1))
        mask = jnp.array([True, True, False])
        hg = hgx.Hypergraph(
            node_features=jnp.ones((3, 2)),
            incidence=H,
            node_mask=mask,
        )
        conv = hgx.HGNNConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg)
        assert jnp.allclose(out[2], jnp.zeros(4))

    def test_differs_from_unigcn(self, tiny_hypergraph, prng_key):
        """HGNNConv (project-after) should differ from UniGCNConv (project-before)."""
        k1, k2 = jax.random.split(prng_key)
        hgnn = hgx.HGNNConv(in_dim=2, out_dim=4, key=k1)
        unigcn = hgx.UniGCNConv(in_dim=2, out_dim=4, key=k1)  # same key = same weights
        out_hgnn = hgnn(tiny_hypergraph)
        out_unigcn = unigcn(tiny_hypergraph)
        # Different computation orders -> different results
        assert not jnp.allclose(out_hgnn, out_unigcn, atol=1e-3)

    def test_gradient_flow(self, tiny_hypergraph, prng_key):
        conv = hgx.HGNNConv(in_dim=2, out_dim=4, key=prng_key)

        def loss_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = jax.grad(loss_fn)(conv)
        assert not jnp.allclose(grads.linear.weight, 0.0)

    def test_jit(self, tiny_hypergraph, prng_key):
        conv = hgx.HGNNConv(in_dim=2, out_dim=4, key=prng_key)
        jit_conv = eqx.filter_jit(conv)
        out = jit_conv(tiny_hypergraph)
        out_eager = conv(tiny_hypergraph)
        assert jnp.allclose(out, out_eager, atol=1e-6)

    def test_hgnn_stack(self, tiny_hypergraph, prng_key):
        """HGNNConv should work within HGNNStack."""
        stack = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.HGNNConv,
            readout_dim=3,
            key=prng_key,
        )
        out = stack(tiny_hypergraph, inference=True)
        assert out.shape == (4, 3)

    def test_symmetric_nodes_get_equal_output(self, prng_key):
        """Nodes with identical features in the same single hyperedge
        should get identical output."""
        hg = hgx.from_incidence(
            jnp.ones((5, 1)),
            node_features=jnp.ones((5, 3)),
        )
        conv = hgx.HGNNConv(in_dim=3, out_dim=4, key=prng_key)
        out = conv(hg)
        for i in range(1, 5):
            assert jnp.allclose(out[0], out[i], atol=1e-6)


class TestHGNNSparseConv:
    """Test sparse HGNN convolution."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        conv = hgx.HGNNSparseConv(in_dim=2, out_dim=8, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_equivalence_with_dense(self, tiny_hypergraph, prng_key):
        """Sparse HGNNConv should match dense HGNNConv."""
        dense = hgx.HGNNConv(in_dim=2, out_dim=4, key=prng_key)
        sparse = hgx.HGNNSparseConv(in_dim=2, out_dim=4, key=prng_key)

        # Copy weights
        sparse = eqx.tree_at(lambda m: m.linear.weight, sparse, dense.linear.weight)
        sparse = eqx.tree_at(lambda m: m.linear.bias, sparse, dense.linear.bias)

        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(tiny_hypergraph)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_sparse_hypergraph_input(self, prng_key):
        """Should work with SparseHypergraph directly."""
        shg = hgx.from_edge_list_sparse(
            [{0, 1, 2}, {1, 2, 3}],
            node_features=jnp.ones((4, 3)),
        )
        conv = hgx.HGNNSparseConv(in_dim=3, out_dim=4, key=prng_key)
        out = conv(shg)
        assert out.shape == (4, 4)

    def test_sparse_matches_dense_on_sparse_input(self, prng_key):
        """Sparse conv on SparseHypergraph should match dense on Hypergraph."""
        edges = [{0, 1, 2}, {1, 2, 3}]
        features = jnp.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3],
            [1.0, 1.0, 0.1],
            [0.0, 0.0, 0.9],
        ])
        hg_dense = hgx.from_edge_list(edges, num_nodes=4, node_features=features)
        hg_sparse = hgx.from_edge_list_sparse(edges, num_nodes=4, node_features=features)

        dense_conv = hgx.HGNNConv(in_dim=3, out_dim=4, key=prng_key)
        sparse_conv = hgx.HGNNSparseConv(in_dim=3, out_dim=4, key=prng_key)

        # Copy weights
        sparse_conv = eqx.tree_at(
            lambda m: m.linear.weight, sparse_conv, dense_conv.linear.weight
        )
        sparse_conv = eqx.tree_at(
            lambda m: m.linear.bias, sparse_conv, dense_conv.linear.bias
        )

        out_dense = dense_conv(hg_dense)
        out_sparse = sparse_conv(hg_sparse)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_hgnn_stack(self, tiny_hypergraph, prng_key):
        """HGNNSparseConv should work within HGNNStack."""
        stack = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.HGNNSparseConv,
            key=prng_key,
        )
        out = stack(tiny_hypergraph, inference=True)
        assert out.shape == (4, 4)


class TestHGNNStackEnhancements:
    """Test residual, layer_norm, and initial_alpha in HGNNStack."""

    def test_residual(self, tiny_hypergraph, prng_key):
        """Residual connections should work when dims match."""
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 8)],  # second layer: 8->8, residual possible
            conv_cls=hgx.HGNNConv,
            residual=True,
            key=prng_key,
        )
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 8)

    def test_residual_no_effect_dim_mismatch(self, tiny_hypergraph, prng_key):
        """Residual should be skipped when dims don't match."""
        model_res = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],  # all dims differ
            conv_cls=hgx.HGNNConv,
            residual=True,
            key=prng_key,
        )
        model_no_res = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.HGNNConv,
            residual=False,
            key=prng_key,
        )
        out_res = model_res(tiny_hypergraph, inference=True)
        out_no = model_no_res(tiny_hypergraph, inference=True)
        # Same key = same weights, no matching dims = same output
        assert jnp.allclose(out_res, out_no, atol=1e-6)

    def test_layer_norm(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.HGNNConv,
            layer_norm=True,
            key=prng_key,
        )
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 4)

    def test_initial_alpha(self, prng_key):
        """Initial residual should blend with x_0."""
        # Use matching dims so alpha kicks in
        features = jnp.ones((4, 8))
        H = jnp.array([[1, 0], [1, 1], [1, 1], [0, 1]], dtype=jnp.float32)
        hg = hgx.from_incidence(H, node_features=features)

        model = hgx.HGNNStack(
            conv_dims=[(8, 8), (8, 8)],
            conv_cls=hgx.HGNNConv,
            initial_alpha=0.5,
            key=prng_key,
        )
        out = model(hg, inference=True)
        assert out.shape == (4, 8)

    def test_residual_plus_layer_norm(self, tiny_hypergraph, prng_key):
        """Residual + LayerNorm together should work."""
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 8)],
            conv_cls=hgx.HGNNConv,
            residual=True,
            layer_norm=True,
            readout_dim=3,
            key=prng_key,
        )
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 3)

    def test_backward_compat_defaults(self, tiny_hypergraph, prng_key):
        """Default args should match original behavior."""
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.UniGCNConv,
            key=prng_key,
        )
        assert model.residual is False
        assert model.layer_norm is False
        assert model.initial_alpha is None
        assert model.norms is None
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 4)

    def test_grad_through_residual_stack(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 8)],
            conv_cls=hgx.HGNNConv,
            residual=True,
            layer_norm=True,
            readout_dim=2,
            key=prng_key,
        )

        def loss_fn(m):
            return jnp.sum(m(tiny_hypergraph, inference=True))

        grads = jax.grad(loss_fn)(model)
        assert not jnp.allclose(grads.convs[0].linear.weight, 0.0)
        assert not jnp.allclose(grads.readout.weight, 0.0)
