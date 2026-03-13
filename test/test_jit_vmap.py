"""Tests for JAX JIT compilation and vmap compatibility.

These are critical — many custom JAX code silently breaks under
tracing. These tests verify the JAX contract is maintained.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import hgx


class TestJIT:
    """Test that all operations work under jax.jit.

    Equinox modules contain arrays and must be JIT-compiled via
    eqx.filter_jit (which partitions params from static structure)
    rather than raw jax.jit.
    """

    def test_unigcn_jit(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)

        @eqx.filter_jit
        def apply(model, hg):
            return model(hg)

        out_eager = conv(tiny_hypergraph)
        out_jit = apply(conv, tiny_hypergraph)
        assert jnp.allclose(out_eager, out_jit, atol=1e-6)

    def test_thnn_jit(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, key=prng_key)

        @eqx.filter_jit
        def apply(model, hg):
            return model(hg)

        out_eager = conv(tiny_hypergraph)
        out_jit = apply(conv, tiny_hypergraph)
        assert jnp.allclose(out_eager, out_jit, atol=1e-6)

    def test_clique_expansion_jit(self, tiny_hypergraph):
        jit_fn = jax.jit(hgx.clique_expansion)
        A = hgx.clique_expansion(tiny_hypergraph)
        A_jit = jit_fn(tiny_hypergraph)
        assert jnp.allclose(A, A_jit)

    def test_laplacian_jit(self, tiny_hypergraph):
        jit_fn = jax.jit(hgx.hypergraph_laplacian)
        L = hgx.hypergraph_laplacian(tiny_hypergraph)
        L_jit = jit_fn(tiny_hypergraph)
        assert jnp.allclose(L, L_jit, atol=1e-6)


class TestGradUnderJIT:
    """Test that grad works under JIT."""

    def test_unigcn_grad_jit(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)

        @jax.jit
        def loss_and_grad(model):
            def loss_fn(m):
                return jnp.sum(m(tiny_hypergraph))
            return jax.grad(loss_fn)(model)

        grads = loss_and_grad(conv)
        assert not jnp.allclose(grads.linear.weight, 0.0)

    def test_thnn_grad_jit(self, tiny_hypergraph, prng_key):
        conv = hgx.THNNConv(in_dim=2, out_dim=4, rank=16, key=prng_key)

        @jax.jit
        def loss_and_grad(model):
            def loss_fn(m):
                return jnp.sum(m(tiny_hypergraph))
            return jax.grad(loss_fn)(model)

        grads = loss_and_grad(conv)
        assert not jnp.allclose(grads.theta.weight, 0.0)
