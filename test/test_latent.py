"""Tests for latent hypergraph dynamics (encode-integrate-decode)."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest

diffrax = pytest.importorskip("diffrax")


@pytest.fixture
def tiny_hg():
    """4-node hypergraph with 2 hyperedges and 8-dim features."""
    return hgx.from_edge_list(
        [(0, 1, 2), (1, 2, 3)],
        node_features=jnp.ones((4, 8)),
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# -----------------------------------------------------------------------
# LatentHypergraphODE
# -----------------------------------------------------------------------


class TestLatentODE:

    def test_latent_ode_output_shape(self, tiny_hg, key):
        model = hgx.LatentHypergraphODE(
            obs_dim=8, latent_dim=4,
            conv_cls=hgx.UniGCNConv,
            key=key,
        )
        out = model(tiny_hg, t0=0.0, t1=1.0)
        assert out.shape == (4, 8)

    def test_latent_ode_grad(self, tiny_hg, key):
        model = hgx.LatentHypergraphODE(
            obs_dim=8, latent_dim=4,
            conv_cls=hgx.UniGCNConv,
            key=key,
        )

        def loss_fn(m):
            return jnp.sum(m(tiny_hg, t0=0.0, t1=0.5))

        grads = eqx.filter_grad(loss_fn)(model)

        # Encoder gradients
        assert not jnp.allclose(grads.encoder.layers[0].weight, 0.0)
        # Dynamics (conv) gradients
        assert not jnp.allclose(
            grads.dynamics.drift.conv.linear.weight, 0.0
        )
        # Decoder gradients
        assert not jnp.allclose(grads.decoder.layers[-1].weight, 0.0)

    def test_latent_ode_jit(self, tiny_hg, key):
        model = hgx.LatentHypergraphODE(
            obs_dim=8, latent_dim=4,
            conv_cls=hgx.UniGCNConv,
            key=key,
        )

        @eqx.filter_jit
        def forward(m, hg):
            return m(hg, t0=0.0, t1=1.0)

        out_jit = forward(model, tiny_hg)
        out_eager = model(tiny_hg, t0=0.0, t1=1.0)

        assert out_jit.shape == (4, 8)
        assert jnp.allclose(out_jit, out_eager, atol=1e-5)


# -----------------------------------------------------------------------
# LatentHypergraphSDE
# -----------------------------------------------------------------------


class TestLatentSDE:

    def test_latent_sde_stochasticity(self, tiny_hg, key):
        model = hgx.LatentHypergraphSDE(
            obs_dim=8, latent_dim=4, num_nodes=4,
            conv_cls=hgx.UniGCNConv,
            sigma_init=0.5,
            dt=0.1,
            key=key,
        )
        k1, k2 = jax.random.split(key)
        out1 = model(tiny_hg, t0=0.0, t1=1.0, key=k1)
        out2 = model(tiny_hg, t0=0.0, t1=1.0, key=k2)

        assert out1.shape == (4, 8)
        assert out2.shape == (4, 8)
        assert not jnp.allclose(out1, out2, atol=1e-3)

    def test_latent_sde_grad(self, tiny_hg, key):
        k1, k2 = jax.random.split(key)
        model = hgx.LatentHypergraphSDE(
            obs_dim=8, latent_dim=4, num_nodes=4,
            conv_cls=hgx.UniGCNConv,
            dt=0.1,
            key=k1,
        )

        def loss_fn(m):
            return jnp.sum(m(tiny_hg, t0=0.0, t1=0.5, key=k2))

        grads = eqx.filter_grad(loss_fn)(model)

        # Encoder gradients
        assert not jnp.allclose(grads.encoder.layers[0].weight, 0.0)
        # Dynamics drift (conv) gradients
        assert not jnp.allclose(
            grads.dynamics.drift.conv.linear.weight, 0.0
        )
        # Diffusion gradients
        assert not jnp.allclose(grads.dynamics.diffusion.log_sigma, 0.0)
        # Decoder gradients
        assert not jnp.allclose(grads.decoder.layers[-1].weight, 0.0)
