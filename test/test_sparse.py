"""Tests for sparse message passing and UniGCNSparseConv."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import hgx
from hgx._sparse import edge_to_vertex, incidence_to_star_expansion, vertex_to_edge


class TestIncidenceToStarExpansion:
    """Test star-expansion index extraction."""

    def test_indices_match_nonzero(self, tiny_hypergraph):
        H = tiny_hypergraph.incidence
        v_idx, e_idx, valid = incidence_to_star_expansion(H)
        num_valid = int(jnp.sum(valid))
        # Should have 6 real nonzero entries
        assert num_valid == 6
        # Every valid pair should correspond to a 1 in H
        for i in range(num_valid):
            assert H[v_idx[i], e_idx[i]] == 1.0

    def test_empty_incidence(self):
        H = jnp.zeros((3, 2))
        v_idx, e_idx, valid = incidence_to_star_expansion(H)
        assert int(jnp.sum(valid)) == 0


class TestSegmentSumOps:
    """Test vertex_to_edge and edge_to_vertex."""

    def test_v2e_matches_dense(self, tiny_hypergraph):
        H = tiny_hypergraph.incidence
        x = tiny_hypergraph.node_features
        v_idx, e_idx, valid = incidence_to_star_expansion(H)

        sparse_e = vertex_to_edge(x, v_idx, e_idx, H.shape[1], valid)
        dense_e = H.T @ x

        assert jnp.allclose(sparse_e, dense_e, atol=1e-6)

    def test_e2v_matches_dense(self, tiny_hypergraph):
        H = tiny_hypergraph.incidence
        e = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 2 edges, dim 2
        v_idx, e_idx, valid = incidence_to_star_expansion(H)

        sparse_v = edge_to_vertex(e, v_idx, e_idx, H.shape[0], valid)
        dense_v = H @ e

        assert jnp.allclose(sparse_v, dense_v, atol=1e-6)


class TestUniGCNSparseConv:
    """Test sparse UniGCN numerical equivalence with dense version."""

    def _make_pair(self, in_dim, out_dim, key, **kwargs):
        """Create dense and sparse conv with shared weights."""
        dense = hgx.UniGCNConv(in_dim=in_dim, out_dim=out_dim, key=key, **kwargs)
        sparse = hgx.UniGCNSparseConv(in_dim=in_dim, out_dim=out_dim, key=key, **kwargs)
        return dense, sparse

    def test_equivalence_normalized(self, tiny_hypergraph, prng_key):
        dense, sparse = self._make_pair(2, 8, prng_key)
        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(tiny_hypergraph)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_equivalence_unnormalized(self, tiny_hypergraph, prng_key):
        dense, sparse = self._make_pair(2, 8, prng_key, normalize=False)
        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(tiny_hypergraph)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_equivalence_no_bias(self, tiny_hypergraph, prng_key):
        dense, sparse = self._make_pair(2, 4, prng_key, use_bias=False)
        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(tiny_hypergraph)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_equivalence_pairwise(self, pairwise_hypergraph, prng_key):
        dense, sparse = self._make_pair(1, 4, prng_key)
        out_dense = dense(pairwise_hypergraph)
        out_sparse = sparse(pairwise_hypergraph)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_equivalence_single_hyperedge(self, single_hyperedge, prng_key):
        dense, sparse = self._make_pair(5, 4, prng_key)
        out_dense = dense(single_hyperedge)
        out_sparse = sparse(single_hyperedge)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_masked_node(self, prng_key):
        H = jnp.ones((3, 1))
        mask = jnp.array([True, True, False])
        hg = hgx.Hypergraph(
            node_features=jnp.ones((3, 2)),
            incidence=H,
            node_mask=mask,
        )
        dense, sparse = self._make_pair(2, 4, prng_key)
        out_dense = dense(hg)
        out_sparse = sparse(hg)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)
        assert jnp.allclose(out_sparse[2], jnp.zeros(4))

    def test_gradient_flow(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNSparseConv(in_dim=2, out_dim=4, key=prng_key)

        def loss_fn(model):
            return jnp.sum(model(tiny_hypergraph))

        grads = jax.grad(loss_fn)(conv)
        assert not jnp.allclose(grads.linear.weight, 0.0)

    def test_jit(self, tiny_hypergraph, prng_key):
        conv = hgx.UniGCNSparseConv(in_dim=2, out_dim=4, key=prng_key)
        jit_conv = eqx.filter_jit(conv)
        out_jit = jit_conv(tiny_hypergraph)
        out_eager = conv(tiny_hypergraph)
        assert jnp.allclose(out_jit, out_eager, atol=1e-6)
