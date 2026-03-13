"""Tests for perturbation prediction module."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._perturbation import (
    in_silico_knockout,
    perturbation_screen,
    PerturbationEncoder,
    PerturbationPredictor,
    train_perturbation_predictor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def gene_hg():
    """6-gene regulatory hypergraph with 3 regulatory modules, dim 8."""
    return hgx.from_edge_list(
        [(0, 1, 2), (2, 3, 4), (4, 5, 0)],
        node_features=jnp.ones((6, 8)),
    )


@pytest.fixture
def predictor(key):
    """PerturbationPredictor: gene_dim=8, hidden=16, 3 fates."""
    return PerturbationPredictor(
        gene_dim=8,
        hidden_dim=16,
        num_fates=3,
        conv_cls=hgx.UniGCNConv,
        key=key,
    )


# ---------------------------------------------------------------------------
# PerturbationEncoder
# ---------------------------------------------------------------------------


class TestPerturbationEncoder:

    def test_output_shape(self, gene_hg, key):
        enc = PerturbationEncoder(conv=hgx.UniGCNConv(8, 16, key=key))
        mask = jnp.array([True, False, False, False, False, False])
        out = enc(gene_hg, mask)
        assert out.shape == (6, 16)

    def test_perturbation_changes_output(self, gene_hg, key):
        enc = PerturbationEncoder(conv=hgx.UniGCNConv(8, 16, key=key))
        no_ko = jnp.zeros(6, dtype=bool)
        ko = jnp.array([True, False, False, False, False, False])
        out_ctrl = enc(gene_hg, no_ko)
        out_ko = enc(gene_hg, ko)
        assert not jnp.allclose(out_ctrl, out_ko)

    def test_all_knockout_zeros_input(self, gene_hg, key):
        """Knocking out all genes should zero the conv input features."""
        # use_bias=False so Linear(0) = 0
        enc = PerturbationEncoder(
            conv=hgx.UniGCNConv(8, 16, use_bias=False, key=key)
        )
        all_ko = jnp.ones(6, dtype=bool)
        out = enc(gene_hg, all_ko)
        assert jnp.allclose(out, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# PerturbationPredictor — forward pass
# ---------------------------------------------------------------------------


class TestPredictorForward:

    def test_output_shapes(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)
        expr, fate = predictor(gene_hg, mask)
        assert expr.shape == (6, 8)
        assert fate.shape == (3,)

    def test_fate_sums_to_one(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)
        _, fate = predictor(gene_hg, mask)
        assert jnp.allclose(jnp.sum(fate), 1.0, atol=1e-5)

    def test_fate_nonnegative(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)
        _, fate = predictor(gene_hg, mask)
        assert jnp.all(fate >= 0.0)

    def test_outputs_finite(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)
        expr, fate = predictor(gene_hg, mask)
        assert jnp.all(jnp.isfinite(expr))
        assert jnp.all(jnp.isfinite(fate))

    def test_different_knockouts_differ(self, gene_hg, predictor):
        m0 = jnp.zeros(6, dtype=bool).at[0].set(True)
        m3 = jnp.zeros(6, dtype=bool).at[3].set(True)
        expr0, _ = predictor(gene_hg, m0)
        expr3, _ = predictor(gene_hg, m3)
        assert not jnp.allclose(expr0, expr3)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:

    def test_predictor_jit(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)

        @eqx.filter_jit
        def forward(m, hg, pmask):
            return m(hg, pmask, inference=True)

        expr_jit, fate_jit = forward(predictor, gene_hg, mask)
        expr_eager, fate_eager = predictor(gene_hg, mask, inference=True)
        assert jnp.allclose(expr_jit, expr_eager, atol=1e-5)
        assert jnp.allclose(fate_jit, fate_eager, atol=1e-5)

    def test_in_silico_knockout_jit(self, gene_hg, predictor):

        @eqx.filter_jit
        def ko(m, hg):
            return in_silico_knockout(m, hg, gene_idx=0)

        expr, fate = ko(predictor, gene_hg)
        assert expr.shape == (6, 8)
        assert fate.shape == (3,)
        assert jnp.all(jnp.isfinite(expr))


# ---------------------------------------------------------------------------
# vmap screen
# ---------------------------------------------------------------------------


class TestVmapScreen:

    def test_screen_shapes(self, gene_hg, predictor):
        indices = jnp.array([0, 1, 2])
        expr, fate = perturbation_screen(predictor, gene_hg, indices)
        assert expr.shape == (3, 6, 8)
        assert fate.shape == (3, 3)

    def test_screen_results_finite(self, gene_hg, predictor):
        indices = jnp.array([0, 2, 4])
        expr, fate = perturbation_screen(predictor, gene_hg, indices)
        assert jnp.all(jnp.isfinite(expr))
        assert jnp.all(jnp.isfinite(fate))

    def test_screen_knockouts_differ(self, gene_hg, predictor):
        indices = jnp.array([0, 3])
        expr, _ = perturbation_screen(predictor, gene_hg, indices)
        assert not jnp.allclose(expr[0], expr[1])

    def test_screen_fate_sums_to_one(self, gene_hg, predictor):
        indices = jnp.array([0, 1, 2, 3])
        _, fate = perturbation_screen(predictor, gene_hg, indices)
        sums = jnp.sum(fate, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_screen_matches_single(self, gene_hg, predictor):
        """Screen results should match individual in_silico_knockout."""
        indices = jnp.array([0, 2])
        expr_batch, fate_batch = perturbation_screen(
            predictor, gene_hg, indices
        )
        expr_0, fate_0 = in_silico_knockout(predictor, gene_hg, 0)
        expr_2, fate_2 = in_silico_knockout(predictor, gene_hg, 2)
        assert jnp.allclose(expr_batch[0], expr_0, atol=1e-5)
        assert jnp.allclose(fate_batch[0], fate_0, atol=1e-5)
        assert jnp.allclose(expr_batch[1], expr_2, atol=1e-5)
        assert jnp.allclose(fate_batch[1], fate_2, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradient:

    def test_grad_flow_encoder(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)

        def loss_fn(m):
            expr, fate = m(gene_hg, mask, inference=True)
            return jnp.sum(expr) + jnp.sum(fate)

        grads = eqx.filter_grad(loss_fn)(predictor)
        assert not jnp.allclose(grads.encoder.conv.linear.weight, 0.0)

    def test_grad_flow_expression_head(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)

        def loss_fn(m):
            expr, _ = m(gene_hg, mask, inference=True)
            return jnp.sum(expr)

        grads = eqx.filter_grad(loss_fn)(predictor)
        assert not jnp.allclose(grads.expression_head.weight, 0.0)

    def test_grad_flow_fate_head(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)

        def loss_fn(m):
            _, fate = m(gene_hg, mask, inference=True)
            return jnp.sum(fate)

        grads = eqx.filter_grad(loss_fn)(predictor)
        assert not jnp.allclose(grads.fate_head.weight, 0.0)

    def test_grad_all_finite(self, gene_hg, predictor):
        mask = jnp.zeros(6, dtype=bool).at[0].set(True)

        def loss_fn(m):
            expr, fate = m(gene_hg, mask, inference=True)
            return jnp.sum(expr) + jnp.sum(fate)

        grads = eqx.filter_grad(loss_fn)(predictor)
        assert jnp.all(jnp.isfinite(grads.encoder.conv.linear.weight))
        assert jnp.all(jnp.isfinite(grads.expression_head.weight))
        assert jnp.all(jnp.isfinite(grads.fate_head.weight))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTraining:

    def test_training_reduces_loss(self, gene_hg, key):
        k1, k2 = jax.random.split(key)
        pred = PerturbationPredictor(
            gene_dim=8,
            hidden_dim=16,
            num_fates=3,
            conv_cls=hgx.UniGCNConv,
            key=k1,
        )

        # Synthetic CROP-seq data: 2 perturbation observations
        masks = jnp.array(
            [
                [True, False, False, False, False, False],
                [False, False, True, False, False, False],
            ]
        )
        expr_targets = jnp.zeros((2, 6, 8))
        fate_targets = jnp.array(
            [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]]
        )

        def compute_loss(m):
            def single(mask, et, ft):
                pe, pf = m(gene_hg, mask, inference=True)
                return jnp.mean((pe - et) ** 2) - jnp.sum(
                    ft * jnp.log(pf + 1e-8)
                )

            return jnp.mean(
                jax.vmap(single)(masks, expr_targets, fate_targets)
            )

        loss_before = compute_loss(pred)

        pred_trained = train_perturbation_predictor(
            pred,
            gene_hg,
            masks,
            (expr_targets, fate_targets),
            key=k2,
            epochs=30,
            lr=1e-3,
        )

        loss_after = compute_loss(pred_trained)
        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# Export accessibility
# ---------------------------------------------------------------------------


class TestExports:

    def test_accessible_from_hgx(self):
        assert hasattr(hgx, "PerturbationEncoder")
        assert hasattr(hgx, "PerturbationPredictor")
        assert hasattr(hgx, "in_silico_knockout")
        assert hasattr(hgx, "perturbation_screen")
        assert hasattr(hgx, "train_perturbation_predictor")
