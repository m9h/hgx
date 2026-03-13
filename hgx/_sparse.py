"""Sparse message-passing utilities for hypergraphs.

Provides index-based (star-expansion) alternatives to dense matrix
multiplication for vertex-hyperedge aggregation, enabling O(nnz) message
passing instead of O(n*m).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def incidence_to_star_expansion(
    H: Float[Array, "n m"],
) -> tuple[Int[Array, " nnz"], Int[Array, " nnz"], Bool[Array, " nnz"]]:
    """Convert a dense incidence matrix to star-expansion index arrays.

    Returns padded arrays of size n*m for JIT compatibility. The validity
    mask distinguishes real membership entries from padding.

    Args:
        H: Binary incidence matrix of shape (n, m).

    Returns:
        Tuple (v_idx, e_idx, valid) where each real pair represents a
        membership relation, and valid[i] is True for real entries.
    """
    n, m = H.shape
    max_nnz = n * m
    v_idx, e_idx = jnp.nonzero(H, size=max_nnz, fill_value=0)
    valid = jnp.arange(max_nnz) < jnp.sum(H > 0)
    return v_idx, e_idx, valid


def vertex_to_edge(
    x: Float[Array, "n d"],
    v_idx: Int[Array, " nnz"],
    e_idx: Int[Array, " nnz"],
    num_edges: int,
    valid: Bool[Array, " nnz"] | None = None,
) -> Float[Array, "m d"]:
    """Aggregate vertex features to hyperedges via segment_sum.

    Computes e_k = sum_{i in e_k} x_i for each hyperedge k.

    Args:
        x: Vertex feature matrix of shape (n, d).
        v_idx: Vertex indices from star expansion.
        e_idx: Hyperedge indices from star expansion.
        num_edges: Total number of hyperedges (m).
        valid: Optional mask to zero out padding entries.

    Returns:
        Aggregated hyperedge features of shape (m, d).
    """
    vals = x[v_idx]
    if valid is not None:
        vals = vals * valid[:, None]
    return jax.ops.segment_sum(vals, e_idx, num_segments=num_edges)


def edge_to_vertex(
    e: Float[Array, "m d"],
    v_idx: Int[Array, " nnz"],
    e_idx: Int[Array, " nnz"],
    num_nodes: int,
    valid: Bool[Array, " nnz"] | None = None,
) -> Float[Array, "n d"]:
    """Aggregate hyperedge features to vertices via segment_sum.

    Computes h_i = sum_{k: i in e_k} e_k for each vertex i.

    Args:
        e: Hyperedge feature matrix of shape (m, d).
        v_idx: Vertex indices from star expansion.
        e_idx: Hyperedge indices from star expansion.
        num_nodes: Total number of vertices (n).
        valid: Optional mask to zero out padding entries.

    Returns:
        Aggregated vertex features of shape (n, d).
    """
    vals = e[e_idx]
    if valid is not None:
        vals = vals * valid[:, None]
    return jax.ops.segment_sum(vals, v_idx, num_segments=num_nodes)
