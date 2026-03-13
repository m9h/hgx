"""Tests for multi-layer HGNNStack model."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest


class TestHGNNStack:
    """Test the HGNNStack multi-layer model."""

    def test_basic_forward(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.UniGCNConv,
            key=prng_key,
        )
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 4)

    def test_with_readout(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.UniGCNConv,
            readout_dim=2,
            key=prng_key,
        )
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 2)

    def test_with_thnn(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.THNNConv,
            conv_kwargs={"rank": 16},
            readout_dim=3,
            key=prng_key,
        )
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 3)

    def test_with_dropout_training(self, tiny_hypergraph, prng_key):
        k1, k2 = jax.random.split(prng_key)
        model = hgx.HGNNStack(
            conv_dims=[(2, 8)],
            conv_cls=hgx.UniGCNConv,
            dropout_rate=0.5,
            key=k1,
        )
        # Should work with key during training
        out = model(tiny_hypergraph, key=k2, inference=False)
        assert out.shape == (4, 8)

    def test_dropout_training_missing_key(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8)],
            conv_cls=hgx.UniGCNConv,
            dropout_rate=0.5,
            key=prng_key,
        )
        # Should raise ValueError if key is not provided during training
        with pytest.raises(ValueError, match="Must provide key for dropout during training."):
            model(tiny_hypergraph, key=None, inference=False)

    def test_dropout_disabled_at_inference(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8)],
            conv_cls=hgx.UniGCNConv,
            dropout_rate=0.5,
            key=prng_key,
        )
        # Should work without key at inference
        out = model(tiny_hypergraph, inference=True)
        assert out.shape == (4, 8)

    def test_grad_through_stack(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.UniGCNConv,
            readout_dim=2,
            key=prng_key,
        )

        def loss_fn(m):
            return jnp.sum(m(tiny_hypergraph, inference=True))

        grads = jax.grad(loss_fn)(model)
        # Check grads exist on first and last layer
        assert not jnp.allclose(grads.convs[0].linear.weight, 0.0)
        assert not jnp.allclose(grads.readout.weight, 0.0)

    def test_jit_stack(self, tiny_hypergraph, prng_key):
        model = hgx.HGNNStack(
            conv_dims=[(2, 8), (8, 4)],
            conv_cls=hgx.UniGCNConv,
            readout_dim=2,
            key=prng_key,
        )

        @eqx.filter_jit
        def apply(m, hg):
            return m(hg, inference=True)

        out_eager = model(tiny_hypergraph, inference=True)
        out_jit = apply(model, tiny_hypergraph)
        assert jnp.allclose(out_eager, out_jit, atol=1e-6)
