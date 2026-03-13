"""Tests for hypergraph convolution layers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import hgx


class TestUniGCNConv:
    """Test UniGCN convolution layer."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=2, out_dim=8, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_output_shape_no_bias(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=2, out_dim=8, use_bias=False, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_unnormalized(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, normalize=False, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 4)
        # Output should not be all zeros
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
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg)
        assert jnp.allclose(out[3], jnp.zeros(4))

    def test_single_hyperedge_symmetry(self, single_hyperedge, prng_key):
        """All nodes in the same single hyperedge get identical output
        when features are identical."""
        features = jnp.ones((5, 3))
        hg = hgx.from_incidence(
            jnp.ones((5, 1)),
            node_features=features,
        )
        conv = hgx.UniGCNConv(in_dim=3, out_dim=4, key=prng_key)
        out = conv(hg)
        # All outputs should be identical since all nodes are symmetric
        for i in range(1, 5):
            assert jnp.allclose(out[0], out[i], atol=1e-6)

    def test_masked_node_gets_zero(self, prng_key):
        """Masked nodes should produce zero output."""
        H = jnp.ones((3, 1))
        mask = jnp.array([True, True, False])
        hg = hgx.Hypergraph(
            node_features=jnp.ones((3, 2)),
            incidence=H,
            node_mask=mask,
        )
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg)
        assert jnp.allclose(out[2], jnp.zeros(4))


class TestTHNNConv:
    """Test Tensorized Hypergraph Neural Network convolution."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNConv(in_dim=2, out_dim=8, rank=16, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_output_shape_high_rank(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNConv(in_dim=2, out_dim=4, rank=128, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 4)

    def test_isolated_node_gets_zero(self, prng_key):
        """Isolated node should get zero output in THNN too."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        features = jnp.ones((4, 2))
        hg = hgx.from_incidence(H, node_features=features)
        conv = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, key=prng_key)
        out = conv(hg)
        assert jnp.allclose(out[3], jnp.zeros(4))

    def test_not_identical_to_unigcn(self, tiny_hypergraph, prng_key):
        """THNN should generally produce different results than UniGCN
        since it captures higher-order interactions."""
        k1, k2 = jax.random.split(prng_key)
        unigcn = hgx.UniGCNConv(in_dim=2, out_dim=4, key=k1)
        thnn = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, key=k2)
        out1 = unigcn(tiny_hypergraph)
        out2 = thnn(tiny_hypergraph)
        # Different architectures, different weights -> different outputs
        assert not jnp.allclose(out1, out2, atol=1e-3)

    def test_unnormalized(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, normalize=False, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 4)
        assert jnp.any(out != 0)


class TestUniGATConv:
    """Test UniGAT attention-based convolution layer."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGATConv(in_dim=2, out_dim=8, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_output_shape_no_bias(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGATConv(in_dim=2, out_dim=8, use_bias=False, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

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
        conv = hgx.UniGATConv(in_dim=2, out_dim=4, key=prng_key)
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
        conv = hgx.UniGATConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg)
        assert jnp.allclose(out[2], jnp.zeros(4))

    def test_jit(self, tiny_hypergraph, prng_key):
        """UniGAT should work under eqx.filter_jit."""
        conv = hgx.UniGATConv(in_dim=2, out_dim=4, key=prng_key)
        jit_conv = eqx.filter_jit(conv)
        out = jit_conv(tiny_hypergraph)
        assert out.shape == (4, 4)
        out_eager = conv(tiny_hypergraph)
        assert jnp.allclose(out, out_eager, atol=1e-6)


class TestUniGINConv:
    """Test UniGIN convolution layer."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGINConv(in_dim=2, out_dim=8, hidden_dim=16, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

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
        conv = hgx.UniGINConv(in_dim=2, out_dim=4, hidden_dim=16, key=prng_key)
        out = conv(hg)
        assert jnp.allclose(out[3], jnp.zeros(4))

    def test_gradients(self, tiny_hypergraph, prng_key):
        """Gradients should flow through MLP and epsilon."""
        conv = hgx.UniGINConv(in_dim=2, out_dim=4, hidden_dim=16, key=prng_key)

        @eqx.filter_grad
        def grad_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = grad_fn(conv)
        # MLP layers should have non-zero gradients
        assert not jnp.allclose(grads.mlp.layers[0].weight, 0.0)
        # Epsilon should have a gradient
        assert grads.epsilon != 0.0

    def test_jit(self, tiny_hypergraph, prng_key):
        """UniGIN should work under eqx.filter_jit."""
        conv = hgx.UniGINConv(in_dim=2, out_dim=4, hidden_dim=16, key=prng_key)
        jit_conv = eqx.filter_jit(conv)
        out = jit_conv(tiny_hypergraph)
        assert out.shape == (4, 4)
        out_eager = conv(tiny_hypergraph)
        assert jnp.allclose(out, out_eager, atol=1e-6)

    def test_hgnn_stack(self, tiny_hypergraph, prng_key):
        """UniGIN should work within HGNNStack."""
        stack = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.UniGINConv,
            conv_kwargs={"hidden_dim": 16},
            key=prng_key,
        )
        out = stack(tiny_hypergraph, inference=True)
        assert out.shape == (4, 4)


class TestTHNNSparseConv:
    """Test sparse THNN convolution layer."""

    def test_output_shape(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNSparseConv(in_dim=2, out_dim=8, rank=16, key=prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 8)

    def test_equivalence_with_dense(self, tiny_hypergraph, prng_key):
        """Sparse THNN should produce the same output as dense THNN."""
        dense = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, key=prng_key)
        sparse = hgx.THNNSparseConv(in_dim=2, out_dim=4, rank=16, key=prng_key)

        # Copy weights from dense to sparse via tree_at
        sparse = eqx.tree_at(lambda m: m.theta, sparse, dense.theta)
        sparse = eqx.tree_at(lambda m: m.q, sparse, dense.q)

        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(tiny_hypergraph)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_equivalence_unnormalized(self, tiny_hypergraph, prng_key):
        """Sparse and dense should also match without normalization."""
        dense = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, normalize=False, key=prng_key)
        sparse = hgx.THNNSparseConv(in_dim=2, out_dim=4, rank=16, normalize=False, key=prng_key)

        sparse = eqx.tree_at(lambda m: m.theta, sparse, dense.theta)
        sparse = eqx.tree_at(lambda m: m.q, sparse, dense.q)

        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(tiny_hypergraph)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_hgnn_stack(self, tiny_hypergraph, prng_key):
        """THNNSparseConv should work within HGNNStack."""
        stack = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.THNNSparseConv,
            conv_kwargs={"rank": 16},
            key=prng_key,
        )
        out = stack(tiny_hypergraph, inference=True)
        assert out.shape == (4, 4)


class TestConvGradients:
    """Test that gradients flow through convolution layers."""

    def test_unigcn_grad(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)

        def loss_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = jax.grad(loss_fn)(conv)
        # Check that linear layer weights have non-zero gradients
        assert not jnp.allclose(grads.linear.weight, 0.0)

    def test_unigat_grad(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGATConv(in_dim=2, out_dim=4, key=prng_key)

        def loss_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = jax.grad(loss_fn)(conv)
        assert not jnp.allclose(grads.linear.weight, 0.0)
        assert not jnp.allclose(grads.attn, 0.0)

    def test_thnn_grad(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, key=prng_key)

        def loss_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = jax.grad(loss_fn)(conv)
        assert not jnp.allclose(grads.theta.weight, 0.0)
        assert not jnp.allclose(grads.q.weight, 0.0)
