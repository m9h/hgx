"""Shared test fixtures for hgx tests."""

import jax
import jax.numpy as jnp
import pytest

import hgx


@pytest.fixture
def tiny_hypergraph():
    """A minimal hypergraph with 4 nodes and 2 hyperedges.

    Hyperedge 0: {0, 1, 2}  (triangle)
    Hyperedge 1: {1, 2, 3}  (overlapping triangle)

    Incidence matrix:
        [[1, 0],
         [1, 1],
         [1, 1],
         [0, 1]]
    """
    H = jnp.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    features = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    return hgx.from_incidence(H, node_features=features)


@pytest.fixture
def pairwise_hypergraph():
    """A 2-uniform hypergraph equivalent to a path graph 0-1-2-3.

    Each edge has exactly 2 vertices, so UniGCN should reduce to GCN.

    Hyperedges: {0,1}, {1,2}, {2,3}
    """
    return hgx.from_edge_list([(0, 1), (1, 2), (2, 3)])


@pytest.fixture
def single_hyperedge():
    """A hypergraph with one hyperedge containing all 5 nodes."""
    H = jnp.ones((5, 1))
    features = jnp.eye(5)  # one-hot features
    return hgx.from_incidence(H, node_features=features)


@pytest.fixture
def prng_key():
    """A fixed PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)
