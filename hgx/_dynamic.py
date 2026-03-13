"""Dynamic topology operations for pre-allocated masked hypergraphs.

All operations are JIT-compatible: array shapes never change. Instead,
boolean masks track which node/hyperedge slots are active. Use
``preallocate`` first to create capacity, then ``add_node``,
``add_hyperedge``, ``remove_node``, ``remove_hyperedge`` to modify
the topology functionally (no mutation).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from hgx._hypergraph import Hypergraph


def preallocate(hg: Hypergraph, max_nodes: int, max_edges: int) -> Hypergraph:
    """Pad a hypergraph to fixed capacity and initialise masks.

    Args:
        hg: Source hypergraph (may or may not already have masks).
        max_nodes: Total node capacity (must be >= current node count).
        max_edges: Total hyperedge capacity (must be >= current edge count).

    Returns:
        A new Hypergraph with padded arrays and boolean masks marking
        only the original entries as active.
    """
    n, d = hg.node_features.shape
    m = hg.incidence.shape[1]

    # Pad node features
    node_features = jnp.zeros((max_nodes, d)).at[:n].set(hg.node_features)

    # Pad incidence
    incidence = jnp.zeros((max_nodes, max_edges)).at[:n, :m].set(hg.incidence)

    # Node mask: respect existing mask if present
    node_mask = jnp.zeros(max_nodes, dtype=bool)
    if hg.node_mask is not None:
        node_mask = node_mask.at[:n].set(hg.node_mask)
    else:
        node_mask = node_mask.at[:n].set(True)

    # Edge mask: respect existing mask if present
    edge_mask = jnp.zeros(max_edges, dtype=bool)
    if hg.edge_mask is not None:
        edge_mask = edge_mask.at[:m].set(hg.edge_mask)
    else:
        edge_mask = edge_mask.at[:m].set(True)

    # Pad positions if present
    positions = None
    if hg.positions is not None:
        s = hg.positions.shape[1]
        positions = jnp.zeros((max_nodes, s)).at[:n].set(hg.positions)

    # Pad edge features if present
    edge_features = None
    if hg.edge_features is not None:
        de = hg.edge_features.shape[1]
        edge_features = jnp.zeros((max_edges, de)).at[:m].set(hg.edge_features)

    return Hypergraph(
        node_features=node_features,
        incidence=incidence,
        edge_features=edge_features,
        positions=positions,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )


def add_node(
    hg: Hypergraph,
    features: Float[Array, " d"],
    hyperedges: Bool[Array, " m"] | None = None,
) -> Hypergraph:
    """Activate the next unused node slot.

    Args:
        hg: Pre-allocated hypergraph (must have node_mask).
        features: Feature vector for the new node.
        hyperedges: Optional boolean array of length max_edges indicating
            which hyperedges the new node belongs to.

    Returns:
        Updated Hypergraph with one more active node.
    """
    # First False position = first unused slot
    idx = jnp.argmin(hg.node_mask)

    node_mask = hg.node_mask.at[idx].set(True)
    node_features = hg.node_features.at[idx].set(features)

    incidence = hg.incidence
    if hyperedges is not None:
        incidence = incidence.at[idx].set(hyperedges.astype(hg.incidence.dtype))

    return Hypergraph(
        node_features=node_features,
        incidence=incidence,
        edge_features=hg.edge_features,
        positions=hg.positions,
        node_mask=node_mask,
        edge_mask=hg.edge_mask,
    )


def add_hyperedge(
    hg: Hypergraph,
    members: Bool[Array, " n"],
    features: Float[Array, " de"] | None = None,
) -> Hypergraph:
    """Activate the next unused hyperedge slot.

    Args:
        hg: Pre-allocated hypergraph (must have edge_mask).
        members: Boolean array of length max_nodes indicating which
            nodes belong to this hyperedge.
        features: Optional feature vector for the new hyperedge.

    Returns:
        Updated Hypergraph with one more active hyperedge.
    """
    idx = jnp.argmin(hg.edge_mask)

    edge_mask = hg.edge_mask.at[idx].set(True)
    incidence = hg.incidence.at[:, idx].set(members.astype(hg.incidence.dtype))

    edge_features = hg.edge_features
    if features is not None and edge_features is not None:
        edge_features = edge_features.at[idx].set(features)

    return Hypergraph(
        node_features=hg.node_features,
        incidence=incidence,
        edge_features=edge_features,
        positions=hg.positions,
        node_mask=hg.node_mask,
        edge_mask=edge_mask,
    )


def remove_node(hg: Hypergraph, idx: int) -> Hypergraph:
    """Deactivate a node and zero its incidence row.

    Args:
        hg: Pre-allocated hypergraph (must have node_mask).
        idx: Index of the node to remove.

    Returns:
        Updated Hypergraph with the node masked out.
    """
    node_mask = hg.node_mask.at[idx].set(False)
    incidence = hg.incidence.at[idx].set(0.0)

    return Hypergraph(
        node_features=hg.node_features,
        incidence=incidence,
        edge_features=hg.edge_features,
        positions=hg.positions,
        node_mask=node_mask,
        edge_mask=hg.edge_mask,
    )


def remove_hyperedge(hg: Hypergraph, idx: int) -> Hypergraph:
    """Deactivate a hyperedge and zero its incidence column.

    Args:
        hg: Pre-allocated hypergraph (must have edge_mask).
        idx: Index of the hyperedge to remove.

    Returns:
        Updated Hypergraph with the hyperedge masked out.
    """
    edge_mask = hg.edge_mask.at[idx].set(False)
    incidence = hg.incidence.at[:, idx].set(0.0)

    return Hypergraph(
        node_features=hg.node_features,
        incidence=incidence,
        edge_features=hg.edge_features,
        positions=hg.positions,
        node_mask=hg.node_mask,
        edge_mask=edge_mask,
    )
