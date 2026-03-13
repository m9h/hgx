"""Tests for hypergraph convolution layers."""

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


class TestConvGradients:
    """Test that gradients flow through convolution layers."""

    def test_unigcn_grad(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)

        def loss_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = jax.grad(loss_fn)(conv)
        # Check that linear layer weights have non-zero gradients
        assert not jnp.allclose(grads.linear.weight, 0.0)

    def test_thnn_grad(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, key=prng_key)

        def loss_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = jax.grad(loss_fn)(conv)
        assert not jnp.allclose(grads.theta.weight, 0.0)
        assert not jnp.allclose(grads.q.weight, 0.0)
