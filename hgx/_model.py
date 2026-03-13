"""Composable multi-layer hypergraph neural network models."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class HGNNStack(eqx.Module):
    """A stack of hypergraph convolution layers with activation and dropout.

    Provides a standard multi-layer architecture:
        for each layer:
            x = activation(conv(hg_with_x))
            x = dropout(x)  (training only)
        x = readout(x)

    Attributes:
        convs: List of hypergraph convolution layers.
        readout: Optional final linear layer for classification/regression.
        activation: Activation function applied after each conv.
        dropout_rate: Dropout probability (applied during training).
    """

    convs: list[AbstractHypergraphConv]
    readout: eqx.nn.Linear | None
    activation: Callable = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(
        self,
        conv_dims: list[tuple[int, int]],
        conv_cls: type[AbstractHypergraphConv],
        readout_dim: int | None = None,
        activation: Callable = jax.nn.relu,
        dropout_rate: float = 0.0,
        conv_kwargs: dict | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize HGNNStack.

        Args:
            conv_dims: List of (in_dim, out_dim) for each conv layer.
            conv_cls: Convolution class to use (e.g., UniGCNConv, THNNConv).
            readout_dim: If given, adds a final Linear(last_out_dim, readout_dim).
            activation: Activation function (default: relu).
            dropout_rate: Dropout rate (0.0 = no dropout).
            conv_kwargs: Extra kwargs passed to each conv constructor.
            key: PRNG key for initialization.
        """
        if conv_kwargs is None:
            conv_kwargs = {}
        keys = jax.random.split(key, len(conv_dims) + 1)
        self.convs = [
            conv_cls(in_d, out_d, key=keys[i], **conv_kwargs)
            for i, (in_d, out_d) in enumerate(conv_dims)
        ]
        if readout_dim is not None:
            last_dim = conv_dims[-1][1]
            self.readout = eqx.nn.Linear(last_dim, readout_dim, key=keys[-1])
        else:
            self.readout = None
        self.activation = activation
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        hg: Hypergraph,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Float[Array, "n out"]:
        """Forward pass through the stack.

        Args:
            hg: Input hypergraph.
            key: PRNG key for dropout (required if dropout_rate > 0 and not inference).
            inference: If True, disables dropout.

        Returns:
            Node-level output features.
        """
        x = hg.node_features

        for i, conv in enumerate(self.convs):
            # Build a hypergraph with updated features
            hg_i = Hypergraph(
                node_features=x,
                incidence=hg.incidence,
                edge_features=hg.edge_features,
                positions=hg.positions,
                node_mask=hg.node_mask,
                edge_mask=hg.edge_mask,
                geometry=hg.geometry,
            )
            x = conv(hg_i)
            x = self.activation(x)

            if self.dropout_rate > 0.0 and not inference:
                if key is None:
                    raise ValueError(
                        "Must provide key for dropout during training."
                    )
                key, subkey = jax.random.split(key)
                mask = jax.random.bernoulli(subkey, 1.0 - self.dropout_rate, x.shape)
                x = x * mask / (1.0 - self.dropout_rate)

        if self.readout is not None:
            x = jax.vmap(self.readout)(x)

        return x
