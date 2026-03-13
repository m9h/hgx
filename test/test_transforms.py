"""Tests for hypergraph transformation utilities."""

import hgx
import jax.numpy as jnp


class TestCliqueExpansion:
    """Test clique expansion (hypergraph -> graph)."""

    def test_basic_clique(self, tiny_hypergraph):
        A = hgx.clique_expansion(tiny_hypergraph)
        assert A.shape == (4, 4)
        # Symmetric
        assert jnp.allclose(A, A.T)
        # No self-loops
        assert jnp.allclose(jnp.diag(A), jnp.zeros(4))

    def test_pairwise_clique_matches_adjacency(self):
        """For a 2-uniform hypergraph, clique expansion should
        recover the original adjacency matrix."""
        # Path graph 0-1-2
        A_orig = jnp.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=jnp.float32)
        hg = hgx.from_adjacency(A_orig)
        A_recovered = hgx.clique_expansion(hg)
        assert jnp.allclose(A_recovered, A_orig)

    def test_triangle_hyperedge_becomes_clique(self):
        """A single 3-vertex hyperedge should expand to K3."""
        hg = hgx.from_edge_list([(0, 1, 2)])
        A = hgx.clique_expansion(hg)
        expected = jnp.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=jnp.float32)
        assert jnp.allclose(A, expected)


class TestHypergraphLaplacian:
    """Test hypergraph Laplacian computation."""

    def test_laplacian_shape(self, tiny_hypergraph):
        L = hgx.hypergraph_laplacian(tiny_hypergraph)
        assert L.shape == (4, 4)

    def test_laplacian_symmetric(self, tiny_hypergraph):
        L = hgx.hypergraph_laplacian(tiny_hypergraph)
        assert jnp.allclose(L, L.T, atol=1e-6)

    def test_laplacian_psd(self, tiny_hypergraph):
        """The normalized Laplacian should be positive semi-definite."""
        L = hgx.hypergraph_laplacian(tiny_hypergraph)
        eigenvalues = jnp.linalg.eigvalsh(L)
        # All eigenvalues >= 0 (within numerical tolerance)
        assert jnp.all(eigenvalues > -1e-6)

    def test_unnormalized_laplacian(self, tiny_hypergraph):
        L = hgx.hypergraph_laplacian(tiny_hypergraph, normalized=False)
        assert L.shape == (4, 4)
        # Row sums of unnormalized Laplacian applied to constant vector = 0
        ones = jnp.ones(4)
        assert jnp.allclose(L @ ones, jnp.zeros(4), atol=1e-6)
