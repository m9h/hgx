"""Abstract base class for hypergraph convolution layers."""

from __future__ import annotations

import abc

import equinox as eqx
from jaxtyping import Array, Float

from hgx._hypergraph import Hypergraph


class AbstractHypergraphConv(eqx.Module):
    """Abstract base class for all hypergraph convolution layers.

    Subclasses implement __call__ to transform node features via
    message passing over the hypergraph structure. The interface
    is designed to accommodate both topology-only and geometry-aware
    layers through the Hypergraph data structure.
    """

    @abc.abstractmethod
    def __call__(
        self,
        hg: Hypergraph,
    ) -> Float[Array, "n out_dim"]:
        """Apply hypergraph convolution to produce updated node features.

        Args:
            hg: Input hypergraph with node features and incidence structure.

        Returns:
            Updated node feature matrix of shape (num_nodes, out_dim).
        """
        ...
