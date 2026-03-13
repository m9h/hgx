"""Tests for dynamic topology operations."""

import equinox as eqx
import hgx
import jax.numpy as jnp
import pytest


@pytest.fixture
def preallocated_hg(tiny_hypergraph):
    """tiny_hypergraph (4 nodes, 2 edges) padded to capacity 8/6."""
    return hgx.preallocate(tiny_hypergraph, max_nodes=8, max_edges=6)


class TestPreallocate:
    """Test that preallocate preserves the original structure."""

    def test_shapes(self, preallocated_hg):
        hg = preallocated_hg
        assert hg.node_features.shape == (8, 2)
        assert hg.incidence.shape == (8, 6)
        assert hg.node_mask.shape == (8,)
        assert hg.edge_mask.shape == (6,)

    def test_active_counts(self, preallocated_hg):
        hg = preallocated_hg
        assert hg.num_nodes == 4
        assert hg.num_edges == 2

    def test_original_features_preserved(self, tiny_hypergraph, preallocated_hg):
        orig = tiny_hypergraph.node_features
        assert jnp.allclose(preallocated_hg.node_features[:4], orig)

    def test_original_incidence_preserved(self, tiny_hypergraph, preallocated_hg):
        orig = tiny_hypergraph.incidence
        assert jnp.allclose(preallocated_hg.incidence[:4, :2], orig)

    def test_padded_slots_are_zero(self, preallocated_hg):
        hg = preallocated_hg
        assert jnp.allclose(hg.node_features[4:], 0.0)
        assert jnp.allclose(hg.incidence[4:, :], 0.0)
        assert jnp.allclose(hg.incidence[:, 2:], 0.0)

    def test_masks_correct(self, preallocated_hg):
        hg = preallocated_hg
        expected_node = jnp.array([True, True, True, True, False, False, False, False])
        expected_edge = jnp.array([True, True, False, False, False, False])
        assert jnp.array_equal(hg.node_mask, expected_node)
        assert jnp.array_equal(hg.edge_mask, expected_edge)

    def test_positions_padded(self):
        """Positions are padded when present."""
        H = jnp.eye(3, 2)
        pos = jnp.ones((3, 2))
        hg = hgx.Hypergraph(
            node_features=jnp.ones((3, 1)),
            incidence=H,
            positions=pos,
        )
        padded = hgx.preallocate(hg, max_nodes=5, max_edges=4)
        assert padded.positions.shape == (5, 2)
        assert jnp.allclose(padded.positions[:3], 1.0)
        assert jnp.allclose(padded.positions[3:], 0.0)


class TestAddNode:
    """Test adding nodes to a preallocated hypergraph."""

    def test_num_nodes_increases(self, preallocated_hg):
        new_feat = jnp.array([5.0, 6.0])
        hg2 = hgx.add_node(preallocated_hg, new_feat)
        assert hg2.num_nodes == 5

    def test_features_correct(self, preallocated_hg):
        new_feat = jnp.array([5.0, 6.0])
        hg2 = hgx.add_node(preallocated_hg, new_feat)
        # New node should be at index 4 (first unused slot)
        assert jnp.allclose(hg2.node_features[4], new_feat)

    def test_with_hyperedges(self, preallocated_hg):
        new_feat = jnp.array([5.0, 6.0])
        # Connect new node to hyperedge 0 only
        edges = jnp.array([True, False, False, False, False, False])
        hg2 = hgx.add_node(preallocated_hg, new_feat, hyperedges=edges)
        assert hg2.incidence[4, 0] == 1.0
        assert hg2.incidence[4, 1] == 0.0

    def test_add_two_nodes(self, preallocated_hg):
        hg2 = hgx.add_node(preallocated_hg, jnp.array([1.0, 2.0]))
        hg3 = hgx.add_node(hg2, jnp.array([3.0, 4.0]))
        assert hg3.num_nodes == 6
        assert jnp.allclose(hg3.node_features[4], jnp.array([1.0, 2.0]))
        assert jnp.allclose(hg3.node_features[5], jnp.array([3.0, 4.0]))


class TestAddHyperedge:
    """Test adding hyperedges to a preallocated hypergraph."""

    def test_num_edges_increases(self, preallocated_hg):
        members = jnp.array([True, True, False, False, False, False, False, False])
        hg2 = hgx.add_hyperedge(preallocated_hg, members)
        assert hg2.num_edges == 3

    def test_incidence_correct(self, preallocated_hg):
        members = jnp.array([True, True, False, False, False, False, False, False])
        hg2 = hgx.add_hyperedge(preallocated_hg, members)
        # New hyperedge at index 2 (first unused slot)
        assert hg2.incidence[0, 2] == 1.0
        assert hg2.incidence[1, 2] == 1.0
        assert hg2.incidence[2, 2] == 0.0


class TestRemoveNode:
    """Test removing nodes from a preallocated hypergraph."""

    def test_num_nodes_decreases(self, preallocated_hg):
        hg2 = hgx.remove_node(preallocated_hg, 0)
        assert hg2.num_nodes == 3

    def test_incidence_zeroed(self, preallocated_hg):
        hg2 = hgx.remove_node(preallocated_hg, 1)
        assert jnp.allclose(hg2.incidence[1], 0.0)

    def test_mask_updated(self, preallocated_hg):
        hg2 = hgx.remove_node(preallocated_hg, 2)
        assert not hg2.node_mask[2]

    def test_features_unchanged(self, preallocated_hg):
        """Features are kept (just masked out), not zeroed."""
        hg2 = hgx.remove_node(preallocated_hg, 0)
        assert jnp.allclose(hg2.node_features[0], preallocated_hg.node_features[0])


class TestRemoveHyperedge:
    """Test removing hyperedges."""

    def test_num_edges_decreases(self, preallocated_hg):
        hg2 = hgx.remove_hyperedge(preallocated_hg, 0)
        assert hg2.num_edges == 1

    def test_incidence_column_zeroed(self, preallocated_hg):
        hg2 = hgx.remove_hyperedge(preallocated_hg, 0)
        assert jnp.allclose(hg2.incidence[:, 0], 0.0)


class TestConvOnDynamic:
    """Convolution still works on dynamically modified hypergraphs."""

    def test_unigcn_after_add_node(self, preallocated_hg, prng_key):
        hg2 = hgx.add_node(preallocated_hg, jnp.array([1.0, 0.0]))
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg2)
        assert out.shape == (8, 4)  # max_nodes x out_dim

    def test_unigcn_after_remove_node(self, preallocated_hg, prng_key):
        hg2 = hgx.remove_node(preallocated_hg, 0)
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg2)
        assert out.shape == (8, 4)
        # Removed node should have zero output
        assert jnp.allclose(out[0], 0.0)

    def test_unigcn_after_add_hyperedge(self, preallocated_hg, prng_key):
        members = jnp.array([True, False, False, True, False, False, False, False])
        hg2 = hgx.add_hyperedge(preallocated_hg, members)
        conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        out = conv(hg2)
        assert out.shape == (8, 4)


class TestJITCompatibility:
    """Dynamic operations must work under eqx.filter_jit."""

    def test_add_node_jit(self, preallocated_hg):
        new_feat = jnp.array([7.0, 8.0])

        @eqx.filter_jit
        def do_add(hg, feat):
            return hgx.add_node(hg, feat)

        hg2 = do_add(preallocated_hg, new_feat)
        assert hg2.num_nodes == 5
        assert jnp.allclose(hg2.node_features[4], new_feat)

    def test_add_hyperedge_jit(self, preallocated_hg):
        members = jnp.array([True, True, True, False, False, False, False, False])

        @eqx.filter_jit
        def do_add(hg, m):
            return hgx.add_hyperedge(hg, m)

        hg2 = do_add(preallocated_hg, members)
        assert hg2.num_edges == 3

    def test_remove_node_jit(self, preallocated_hg):
        @eqx.filter_jit
        def do_remove(hg):
            return hgx.remove_node(hg, 0)

        hg2 = do_remove(preallocated_hg)
        assert hg2.num_nodes == 3


class TestRoundTrip:
    """Add then remove should yield an equivalent masked structure."""

    def test_add_then_remove_node(self, preallocated_hg):
        hg2 = hgx.add_node(preallocated_hg, jnp.array([9.0, 9.0]))
        assert hg2.num_nodes == 5
        hg3 = hgx.remove_node(hg2, 4)
        assert hg3.num_nodes == 4
        # Masks should match original
        assert jnp.array_equal(hg3.node_mask, preallocated_hg.node_mask)

    def test_add_then_remove_hyperedge(self, preallocated_hg):
        members = jnp.array([True, True, False, False, False, False, False, False])
        hg2 = hgx.add_hyperedge(preallocated_hg, members)
        assert hg2.num_edges == 3
        hg3 = hgx.remove_hyperedge(hg2, 2)
        assert hg3.num_edges == 2
        assert jnp.array_equal(hg3.edge_mask, preallocated_hg.edge_mask)

    def test_masked_incidence_matches_after_round_trip(self, preallocated_hg):
        """After add+remove, _masked_incidence should be equivalent."""
        hg2 = hgx.add_node(
            preallocated_hg,
            jnp.array([1.0, 1.0]),
            hyperedges=jnp.array([True, False, False, False, False, False]),
        )
        hg3 = hgx.remove_node(hg2, 4)
        orig_H = preallocated_hg._masked_incidence()
        final_H = hg3._masked_incidence()
        assert jnp.allclose(orig_H, final_H)
