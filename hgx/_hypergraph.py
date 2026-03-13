"""Core hypergraph data structure.

A hypergraph generalizes a graph by allowing edges (hyperedges) to connect
any number of vertices simultaneously. This module provides the foundational
data structure for hgx, designed to be forward-compatible with combinatorial
complexes and geometric embeddings while keeping the common case simple.

The representation uses an incidence matrix H where H[i,k] = 1 means
vertex i belongs to hyperedge k. For message passing, we precompute the
"star expansion" — a bipartite graph between vertices and hyperedges —
stored as sender/receiver index arrays for efficient use with
jax.ops.segment_sum.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class Hypergraph(eqx.Module):
    """A hypergraph with vertex features and optional geometry.

    Attributes:
        node_features: Feature matrix of shape (num_nodes, node_dim).
        incidence: Binary incidence matrix of shape (num_nodes, num_edges).
            H[i, k] = 1 means vertex i belongs to hyperedge k.
        edge_features: Optional feature matrix of shape (num_edges, edge_dim).
        positions: Optional spatial coordinates of shape (num_nodes, spatial_dim).
            Enables geometry-aware message passing layers.
        node_mask: Boolean mask of shape (num_nodes,). True for active nodes.
            Supports dynamic topology via pre-allocated arrays.
        edge_mask: Boolean mask of shape (num_edges,). True for active hyperedges.
    """

    node_features: Float[Array, "n d"]
    incidence: Float[Array, "n m"]
    edge_features: Float[Array, "m de"] | None = None
    positions: Float[Array, "n s"] | None = None
    node_mask: Bool[Array, " n"] | None = None
    edge_mask: Bool[Array, " m"] | None = None

    @property
    def num_nodes(self) -> int:
        """Number of (active) nodes."""
        if self.node_mask is not None:
            return int(jnp.sum(self.node_mask))
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of (active) hyperedges."""
        if self.edge_mask is not None:
            return int(jnp.sum(self.edge_mask))
        return self.incidence.shape[1]

    @property
    def node_dim(self) -> int:
        """Dimensionality of node features."""
        return self.node_features.shape[-1]

    @property
    def max_nodes(self) -> int:
        """Total allocated node capacity (including masked)."""
        return self.node_features.shape[0]

    @property
    def max_edges(self) -> int:
        """Total allocated hyperedge capacity (including masked)."""
        return self.incidence.shape[1]

    @property
    def node_degrees(self) -> Float[Array, " n"]:
        """Degree of each node (number of hyperedges it belongs to)."""
        H = self._masked_incidence()
        return jnp.sum(H, axis=1)

    @property
    def edge_degrees(self) -> Float[Array, " m"]:
        """Degree of each hyperedge (number of vertices it contains)."""
        H = self._masked_incidence()
        return jnp.sum(H, axis=0)

    def _masked_incidence(self) -> Float[Array, "n m"]:
        """Return incidence matrix with masks applied."""
        H = self.incidence
        if self.node_mask is not None:
            H = H * self.node_mask[:, None]
        if self.edge_mask is not None:
            H = H * self.edge_mask[None, :]
        return H

    def star_expansion(self) -> tuple[Int[Array, " nnz"], Int[Array, " nnz"]]:
        """Compute star expansion indices for segment_sum message passing.

        Returns a tuple (vertex_indices, hyperedge_indices) where each pair
        (vertex_indices[i], hyperedge_indices[i]) represents a membership
        relation in the bipartite vertex-hyperedge graph.

        Returns:
            Tuple of (vertex_indices, hyperedge_indices) arrays.
        """
        H = self._masked_incidence()
        v_idx, e_idx = jnp.nonzero(H, size=int(jnp.sum(H > 0)))
        return v_idx, e_idx


def from_incidence(
    incidence: Float[Array, "n m"],
    node_features: Float[Array, "n d"] | None = None,
    edge_features: Float[Array, "m de"] | None = None,
    positions: Float[Array, "n s"] | None = None,
) -> Hypergraph:
    """Create a Hypergraph from an incidence matrix.

    Args:
        incidence: Binary incidence matrix H of shape (n, m).
        node_features: Optional node features. Defaults to ones of dim 1.
        edge_features: Optional hyperedge features.
        positions: Optional spatial coordinates for nodes.

    Returns:
        A Hypergraph instance.
    """
    n, m = incidence.shape
    if node_features is None:
        node_features = jnp.ones((n, 1))
    return Hypergraph(
        node_features=node_features,
        incidence=incidence,
        edge_features=edge_features,
        positions=positions,
    )


def from_edge_list(
    edges: list[tuple[int, ...] | list[int]],
    num_nodes: int | None = None,
    node_features: Float[Array, "n d"] | None = None,
) -> Hypergraph:
    """Create a Hypergraph from a list of hyperedges.

    Each hyperedge is a tuple/list of vertex indices.

    Args:
        edges: List of hyperedges, each a sequence of vertex indices.
        num_nodes: Total number of nodes. Inferred from edges if not given.
        node_features: Optional node features.

    Returns:
        A Hypergraph instance.
    """
    if num_nodes is None:
        num_nodes = max(v for e in edges for v in e) + 1
    num_edges = len(edges)
    H = jnp.zeros((num_nodes, num_edges))
    for k, edge in enumerate(edges):
        for v in edge:
            H = H.at[v, k].set(1.0)
    return from_incidence(H, node_features=node_features)


def from_adjacency(
    adjacency: Float[Array, "n n"],
    node_features: Float[Array, "n d"] | None = None,
) -> Hypergraph:
    """Create a Hypergraph from a standard graph adjacency matrix.

    Each edge (i, j) becomes a 2-uniform hyperedge {i, j}.
    Only the upper triangle is used (undirected graph assumed).

    Args:
        adjacency: Adjacency matrix of shape (n, n).
        node_features: Optional node features.

    Returns:
        A Hypergraph with 2-uniform hyperedges.
    """
    rows, cols = jnp.nonzero(jnp.triu(adjacency, k=1), size=int(jnp.sum(jnp.triu(adjacency, k=1) > 0)))
    num_nodes = adjacency.shape[0]
    num_edges = rows.shape[0]
    H = jnp.zeros((num_nodes, num_edges))
    for k in range(num_edges):
        H = H.at[int(rows[k]), k].set(1.0)
        H = H.at[int(cols[k]), k].set(1.0)
    return from_incidence(H, node_features=node_features)
