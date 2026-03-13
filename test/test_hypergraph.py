"""Tests for the core Hypergraph data structure."""

import hgx
import jax.numpy as jnp


class TestHypergraphConstruction:
    """Test creating hypergraphs via different constructors."""

    def test_from_incidence_basic(self, tiny_hypergraph):
        hg = tiny_hypergraph
        assert hg.max_nodes == 4
        assert hg.max_edges == 2
        assert hg.node_dim == 2

    def test_from_incidence_default_features(self):
        H = jnp.eye(3, 2)
        hg = hgx.from_incidence(H)
        assert hg.node_features.shape == (3, 1)
        assert jnp.allclose(hg.node_features, jnp.ones((3, 1)))

    def test_from_edge_list(self, pairwise_hypergraph):
        hg = pairwise_hypergraph
        assert hg.max_nodes == 4
        assert hg.max_edges == 3
        # Check incidence structure
        H = hg.incidence
        # Edge {0, 1} -> column 0 should have 1s at rows 0 and 1
        assert H[0, 0] == 1.0
        assert H[1, 0] == 1.0
        assert H[2, 0] == 0.0

    def test_from_edge_list_explicit_num_nodes(self):
        hg = hgx.from_edge_list([(0, 1)], num_nodes=5)
        assert hg.max_nodes == 5

    def test_from_adjacency(self):
        A = jnp.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=jnp.float32)
        hg = hgx.from_adjacency(A)
        assert hg.max_nodes == 3
        assert hg.max_edges == 2  # edges: (0,1) and (1,2)


class TestHypergraphProperties:
    """Test computed properties of hypergraphs."""

    def test_node_degrees(self, tiny_hypergraph):
        hg = tiny_hypergraph
        d = hg.node_degrees
        # Node 0: in edge 0 only -> degree 1
        # Node 1: in edges 0, 1 -> degree 2
        # Node 2: in edges 0, 1 -> degree 2
        # Node 3: in edge 1 only -> degree 1
        assert jnp.allclose(d, jnp.array([1.0, 2.0, 2.0, 1.0]))

    def test_edge_degrees(self, tiny_hypergraph):
        hg = tiny_hypergraph
        d = hg.edge_degrees
        # Edge 0: {0, 1, 2} -> degree 3
        # Edge 1: {1, 2, 3} -> degree 3
        assert jnp.allclose(d, jnp.array([3.0, 3.0]))

    def test_single_hyperedge_degrees(self, single_hyperedge):
        hg = single_hyperedge
        assert jnp.allclose(hg.node_degrees, jnp.ones(5))
        assert jnp.allclose(hg.edge_degrees, jnp.array([5.0]))


class TestHypergraphMasking:
    """Test masked (dynamic topology) hypergraphs."""

    def test_node_mask(self, tiny_hypergraph):
        hg = tiny_hypergraph
        mask = jnp.array([True, True, False, False])
        hg_masked = hgx.Hypergraph(
            node_features=hg.node_features,
            incidence=hg.incidence,
            node_mask=mask,
        )
        assert hg_masked.num_nodes == 2
        assert hg_masked.max_nodes == 4

    def test_masked_incidence(self):
        H = jnp.ones((3, 2))
        mask = jnp.array([True, True, False])
        hg = hgx.Hypergraph(
            node_features=jnp.ones((3, 1)),
            incidence=H,
            node_mask=mask,
        )
        H_masked = hg._masked_incidence()
        # Row 2 should be zeroed out
        assert jnp.allclose(H_masked[2], jnp.zeros(2))
        assert jnp.allclose(H_masked[0], jnp.ones(2))


class TestStarExpansion:
    """Test star expansion (bipartite index computation)."""

    def test_star_expansion_indices(self, tiny_hypergraph):
        hg = tiny_hypergraph
        v_idx, e_idx = hg.star_expansion()
        # Should have 6 entries: 3 in edge 0, 3 in edge 1
        assert v_idx.shape[0] == 6
        assert e_idx.shape[0] == 6
