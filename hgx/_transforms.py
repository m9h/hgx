"""Transformations between graph and hypergraph representations.

Provides utilities for converting between standard graphs and hypergraphs,
including clique expansion (hypergraph -> graph) and star expansion
(hypergraph -> bipartite graph).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from hgx._hypergraph import Hypergraph


def clique_expansion(hg: Hypergraph) -> Float[Array, "n n"]:
    """Convert a hypergraph to a graph via clique expansion.

    Each hyperedge is expanded into a clique (complete subgraph) among
    its member vertices. The resulting adjacency matrix is:
        A = H @ H^T - diag(d_v)
    where d_v is the vertex degree vector.

    Note: clique expansion loses higher-order interaction information.
    Two vertices appearing together in a large hyperedge are treated
    identically to two vertices sharing many small hyperedges.

    Args:
        hg: Input hypergraph.

    Returns:
        Adjacency matrix of shape (n, n).
    """
    H = hg._masked_incidence()
    A = H @ H.T
    # Remove self-loops
    A = A - jnp.diag(jnp.diag(A))
    # Binarize (multiple shared hyperedges -> single edge)
    A = jnp.where(A > 0, 1.0, 0.0)
    return A


def hypergraph_laplacian(
    hg: Hypergraph,
    normalized: bool = True,
) -> Float[Array, "n n"]:
    """Compute the hypergraph Laplacian matrix.

    The unnormalized Laplacian is:
        L = D_v - H @ D_e^{-1} @ H^T

    The symmetric normalized Laplacian is:
        L_sym = I - D_v^{-1/2} @ H @ D_e^{-1} @ H^T @ D_v^{-1/2}

    Args:
        hg: Input hypergraph.
        normalized: If True, return the symmetric normalized Laplacian.

    Returns:
        Laplacian matrix of shape (n, n).
    """
    H = hg._masked_incidence()
    n = H.shape[0]

    d_v = jnp.sum(H, axis=1)  # vertex degrees
    d_e = jnp.sum(H, axis=0)  # hyperedge degrees

    D_e_inv = jnp.diag(jnp.where(d_e > 0, 1.0 / d_e, 0.0))

    if normalized:
        d_v_inv_sqrt = jnp.where(d_v > 0, 1.0 / jnp.sqrt(d_v), 0.0)
        D_v_inv_sqrt = jnp.diag(d_v_inv_sqrt)
        L = jnp.eye(n) - D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
    else:
        D_v = jnp.diag(d_v)
        L = D_v - H @ D_e_inv @ H.T

    return L
