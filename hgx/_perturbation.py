"""Perturbation prediction for gene regulatory hypergraphs.

Predicts the effects of gene perturbations (knockouts/knockdowns) on a
regulatory hypergraph, inspired by CROP-seq experimental design.

The pipeline:
    1. **PerturbationEncoder** zeros out perturbed-gene features and
       propagates the perturbation signal through the hypergraph.
    2. **PerturbationPredictor** wraps the encoder with an HGNNStack
       and two readout heads — one for per-gene expression changes
       (regression) and one for cell-fate composition shifts
       (classification over K fates).
    3. Convenience functions **in_silico_knockout** and
       **perturbation_screen** run single or batched knockouts.
    4. **train_perturbation_predictor** fits the model to CROP-seq
       observations using optax.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph
from hgx._model import HGNNStack


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class PerturbationEncoder(eqx.Module):
    """Encode a gene perturbation as a modification to node features.

    Zeros out the features of perturbed nodes (simulating a knockout),
    then propagates the perturbation signal through the hypergraph via
    a single convolution layer so that downstream genes "see" the
    disrupted input.

    Attributes:
        conv: Any hgx convolution layer (e.g. ``UniGCNConv``).
    """

    conv: eqx.Module

    def __call__(
        self,
        hg: Hypergraph,
        perturbation_mask: Bool[Array, " n"],
    ) -> Float[Array, "n hidden_dim"]:
        """Apply perturbation and propagate.

        Args:
            hg: Regulatory hypergraph with gene-expression features.
            perturbation_mask: Boolean array — ``True`` for each
                knocked-out gene.

        Returns:
            Encoded node features of shape ``(n, hidden_dim)``.
        """
        # Zero out perturbed genes
        scale = (~perturbation_mask).astype(hg.node_features.dtype)
        x = hg.node_features * scale[:, None]

        hg_pert = Hypergraph(
            node_features=x,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry=hg.geometry,
        )

        return self.conv(hg_pert)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class PerturbationPredictor(eqx.Module):
    """Full perturbation-prediction model.

    Architecture::

        perturbation_mask
              │
              ▼
        PerturbationEncoder  (zero-out + conv)
              │
              ▼
        HGNNStack            (multi-layer message passing)
              │
        ┌─────┴──────┐
        ▼             ▼
    expression_head  fate_head
    (per-node Δexpr)  (mean-pool → softmax)

    Attributes:
        encoder: Perturbation encoder (conv layer).
        stack: Multi-layer HGNN backbone.
        expression_head: Linear predicting per-gene expression changes.
        fate_head: Linear predicting cell-fate probabilities after
            mean-pooling over nodes.
    """

    encoder: PerturbationEncoder
    stack: HGNNStack
    expression_head: eqx.nn.Linear
    fate_head: eqx.nn.Linear

    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int,
        num_fates: int,
        conv_cls: type[AbstractHypergraphConv],
        num_layers: int = 2,
        dropout_rate: float = 0.0,
        conv_kwargs: dict | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize PerturbationPredictor.

        Args:
            gene_dim: Dimensionality of gene-expression features.
            hidden_dim: Hidden dimension throughout the model.
            num_fates: Number of cell-fate classes.
            conv_cls: Convolution class (e.g. ``UniGCNConv``).
            num_layers: Number of message-passing layers in the stack.
            dropout_rate: Dropout probability (training only).
            conv_kwargs: Extra kwargs forwarded to each conv constructor.
            key: PRNG key for initialisation.
        """
        k_enc, k_stack, k_expr, k_fate = jax.random.split(key, 4)
        if conv_kwargs is None:
            conv_kwargs = {}

        self.encoder = PerturbationEncoder(
            conv=conv_cls(gene_dim, hidden_dim, key=k_enc, **conv_kwargs),
        )
        self.stack = HGNNStack(
            conv_dims=[(hidden_dim, hidden_dim)] * num_layers,
            conv_cls=conv_cls,
            dropout_rate=dropout_rate,
            conv_kwargs=conv_kwargs,
            key=k_stack,
        )
        self.expression_head = eqx.nn.Linear(hidden_dim, gene_dim, key=k_expr)
        self.fate_head = eqx.nn.Linear(hidden_dim, num_fates, key=k_fate)

    def __call__(
        self,
        hg: Hypergraph,
        perturbation_mask: Bool[Array, " n"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> tuple[Float[Array, "n gene_dim"], Float[Array, " num_fates"]]:
        """Predict perturbation effects.

        Args:
            hg: Regulatory hypergraph.
            perturbation_mask: Boolean mask (``True`` = knocked-out gene).
            key: PRNG key for dropout (only needed when
                ``inference=False``).
            inference: If ``True``, disables dropout.

        Returns:
            ``(expression_changes, fate_probabilities)`` where
            *expression_changes* has shape ``(n, gene_dim)`` and
            *fate_probabilities* has shape ``(num_fates,)`` summing
            to 1.
        """
        # Encode perturbation
        x = self.encoder(hg, perturbation_mask)

        # Propagate through HGNN stack
        hg_enc = Hypergraph(
            node_features=x,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry=hg.geometry,
        )
        h = self.stack(hg_enc, key=key, inference=inference)

        # Per-node expression change prediction
        expr_changes = jax.vmap(self.expression_head)(h)

        # Global fate prediction via mask-aware mean pooling
        if hg.node_mask is not None:
            w = hg.node_mask.astype(h.dtype)
            h_pooled = jnp.sum(h * w[:, None], axis=0) / jnp.maximum(
                jnp.sum(w), 1.0
            )
        else:
            h_pooled = jnp.mean(h, axis=0)

        fate_logits = self.fate_head(h_pooled)
        fate_probs = jax.nn.softmax(fate_logits)

        return expr_changes, fate_probs


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def in_silico_knockout(
    predictor: PerturbationPredictor,
    hg: Hypergraph,
    gene_idx: int,
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, "n gene_dim"], Float[Array, " num_fates"]]:
    """Knock out a single gene and predict effects.

    Args:
        predictor: Trained perturbation predictor.
        hg: Regulatory hypergraph.
        gene_idx: Index of the gene to knock out.
        key: Optional PRNG key (unused in inference mode).

    Returns:
        ``(expression_changes, fate_probabilities)``.
    """
    n = hg.node_features.shape[0]
    mask = jnp.zeros(n, dtype=bool).at[gene_idx].set(True)
    return predictor(hg, mask, key=key, inference=True)


def perturbation_screen(
    predictor: PerturbationPredictor,
    hg: Hypergraph,
    gene_indices: Float[Array, " K"],
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, "K n gene_dim"], Float[Array, "K num_fates"]]:
    """Screen multiple single-gene knockouts via ``jax.vmap``.

    Args:
        predictor: Trained perturbation predictor.
        hg: Regulatory hypergraph.
        gene_indices: 1-D integer array of gene indices to knock out.
        key: Optional PRNG key.

    Returns:
        ``(expression_changes, fate_probabilities)`` batched over
        knockouts — shapes ``(K, n, gene_dim)`` and ``(K, num_fates)``.
    """
    n = hg.node_features.shape[0]
    masks = jnp.eye(n, dtype=bool)[gene_indices]  # (K, n)

    def _predict_single(mask):
        return predictor(hg, mask, key=key, inference=True)

    return jax.vmap(_predict_single)(masks)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_perturbation_predictor(
    predictor: PerturbationPredictor,
    hg: Hypergraph,
    perturbations: Bool[Array, "P n"],
    targets: tuple[Float[Array, "P n gene_dim"], Float[Array, "P num_fates"]],
    *,
    key: PRNGKeyArray,
    epochs: int = 100,
    lr: float = 1e-3,
) -> PerturbationPredictor:
    """Train the predictor on observed CROP-seq data.

    Uses an Adam optimiser (via *optax*) with a combined loss:
    ``MSE(expression) + cross-entropy(fate)``.

    Args:
        predictor: Untrained predictor.
        hg: Regulatory hypergraph shared across all perturbations.
        perturbations: ``(P, n)`` boolean masks — one per observation.
        targets: Tuple ``(expr_targets, fate_targets)`` where
            *expr_targets* is ``(P, n, gene_dim)`` observed expression
            changes and *fate_targets* is ``(P, num_fates)`` observed
            fate distributions.
        key: PRNG key for training randomness.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        The trained predictor.
    """
    import optax  # optional dep, imported lazily

    expr_targets, fate_targets = targets
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(predictor, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, step_key):
        def loss_fn(m):
            def single_loss(mask, expr_tgt, fate_tgt):
                pred_expr, pred_fate = m(hg, mask, inference=True)
                expr_loss = jnp.mean((pred_expr - expr_tgt) ** 2)
                fate_loss = -jnp.sum(
                    fate_tgt * jnp.log(pred_fate + 1e-8)
                )
                return expr_loss + fate_loss

            losses = jax.vmap(single_loss)(
                perturbations, expr_targets, fate_targets
            )
            return jnp.mean(losses)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    for _ in range(epochs):
        key, subkey = jax.random.split(key)
        predictor, opt_state, _ = step(predictor, opt_state, subkey)

    return predictor
