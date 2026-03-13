"""Tests for sparse incidence matrix support."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
from hgx._sparse_incidence import (
    from_edge_list_sparse,
    from_sparse_incidence,
    sparse_edge_to_vertex,
    sparse_vertex_to_edge,
    SparseHypergraph,
    SparseUniGCNConv,
    to_sparse,
)


# ---------------------------------------------------------------------------
# Round-trip conversion
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test dense -> sparse -> dense preserves the hypergraph."""

    def test_roundtrip_incidence(self, tiny_hypergraph):
        sparse_hg = to_sparse(tiny_hypergraph)
        recovered = sparse_hg.to_dense()
        assert jnp.allclose(recovered.incidence, tiny_hypergraph.incidence)

    def test_roundtrip_node_features(self, tiny_hypergraph):
        sparse_hg = to_sparse(tiny_hypergraph)
        recovered = sparse_hg.to_dense()
        assert jnp.allclose(recovered.node_features, tiny_hypergraph.node_features)

    def test_roundtrip_with_masks(self):
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        features = jnp.ones((3, 2))
        node_mask = jnp.array([True, True, False])
        edge_mask = jnp.array([True, False])
        hg = hgx.Hypergraph(
            node_features=features,
            incidence=H,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )
        sparse_hg = to_sparse(hg)
        recovered = sparse_hg.to_dense()
        assert jnp.allclose(recovered.incidence, H)
        assert jnp.array_equal(recovered.node_mask, node_mask)
        assert jnp.array_equal(recovered.edge_mask, edge_mask)

    def test_roundtrip_edge_features(self):
        H = jnp.array([[1.0, 0.0], [1.0, 1.0]])
        nf = jnp.ones((2, 3))
        ef = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        hg = hgx.Hypergraph(node_features=nf, incidence=H, edge_features=ef)
        sparse_hg = to_sparse(hg)
        recovered = sparse_hg.to_dense()
        assert jnp.allclose(recovered.edge_features, ef)

    def test_roundtrip_pairwise(self, pairwise_hypergraph):
        sparse_hg = to_sparse(pairwise_hypergraph)
        recovered = sparse_hg.to_dense()
        assert jnp.allclose(recovered.incidence, pairwise_hypergraph.incidence)

    def test_roundtrip_single_hyperedge(self, single_hyperedge):
        sparse_hg = to_sparse(single_hyperedge)
        recovered = sparse_hg.to_dense()
        assert jnp.allclose(recovered.incidence, single_hyperedge.incidence)


# ---------------------------------------------------------------------------
# Sparse message-passing primitives
# ---------------------------------------------------------------------------


class TestSparseMessagePassing:
    """Test that sparse V->E and E->V match dense equivalents."""

    def test_v2e_matches_dense(self, tiny_hypergraph):
        H = tiny_hypergraph.incidence
        x = tiny_hypergraph.node_features
        n, m = H.shape

        sparse_hg = to_sparse(tiny_hypergraph)
        sparse_e = sparse_vertex_to_edge(
            x, sparse_hg.indices, sparse_hg.data, sparse_hg.shape
        )
        dense_e = H.T @ x

        assert jnp.allclose(sparse_e, dense_e, atol=1e-6)

    def test_e2v_matches_dense(self, tiny_hypergraph):
        H = tiny_hypergraph.incidence
        e = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        n, m = H.shape

        sparse_hg = to_sparse(tiny_hypergraph)
        sparse_v = sparse_edge_to_vertex(
            e, sparse_hg.indices, sparse_hg.data, sparse_hg.shape
        )
        dense_v = H @ e

        assert jnp.allclose(sparse_v, dense_v, atol=1e-6)

    def test_v2e_with_weighted_data(self):
        """Test V->E when incidence values are not just 0/1."""
        indices = jnp.array([[0, 0], [1, 0], [1, 1]], dtype=jnp.int32)
        data = jnp.array([2.0, 3.0, 1.0])
        shape = (3, 2)
        x = jnp.array([[1.0], [2.0], [3.0]])

        # Dense equivalent
        H = jnp.zeros(shape).at[0, 0].set(2.0).at[1, 0].set(3.0).at[1, 1].set(1.0)
        dense_result = H.T @ x
        sparse_result = sparse_vertex_to_edge(x, indices, data, shape)

        assert jnp.allclose(sparse_result, dense_result, atol=1e-6)


# ---------------------------------------------------------------------------
# SparseUniGCNConv
# ---------------------------------------------------------------------------


class TestSparseUniGCNConv:
    """Test SparseUniGCNConv matches dense UniGCNConv output."""

    def _make_conv(self, in_dim, out_dim, key, **kwargs):
        return SparseUniGCNConv(
            in_dim=in_dim, out_dim=out_dim, key=key, **kwargs
        )

    def _make_pair(self, in_dim, out_dim, key, **kwargs):
        """Create dense UniGCNConv and SparseUniGCNConv with shared weights."""
        dense = hgx.UniGCNConv(in_dim=in_dim, out_dim=out_dim, key=key, **kwargs)
        sparse = SparseUniGCNConv(
            in_dim=in_dim, out_dim=out_dim, key=key, **kwargs
        )
        return dense, sparse

    def test_matches_dense_normalized(self, tiny_hypergraph, prng_key):
        dense, sparse = self._make_pair(2, 8, prng_key)
        sparse_hg = to_sparse(tiny_hypergraph)
        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(sparse_hg)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_matches_dense_unnormalized(self, tiny_hypergraph, prng_key):
        dense, sparse = self._make_pair(2, 8, prng_key, normalize=False)
        sparse_hg = to_sparse(tiny_hypergraph)
        out_dense = dense(tiny_hypergraph)
        out_sparse = sparse(sparse_hg)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)

    def test_accepts_dense_hypergraph(self, tiny_hypergraph, prng_key):
        """SparseUniGCNConv should also work when given a dense Hypergraph."""
        conv = self._make_conv(2, 4, prng_key)
        out = conv(tiny_hypergraph)
        assert out.shape == (4, 4)

    def test_masked_node(self, prng_key):
        H = jnp.ones((3, 1))
        mask = jnp.array([True, True, False])
        hg = hgx.Hypergraph(
            node_features=jnp.ones((3, 2)),
            incidence=H,
            node_mask=mask,
        )
        dense_conv = hgx.UniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        sparse_conv = SparseUniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        sparse_hg = to_sparse(hg)
        out_dense = dense_conv(hg)
        out_sparse = sparse_conv(sparse_hg)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)
        assert jnp.allclose(out_sparse[2], jnp.zeros(4))

    def test_gradient_flow(self, tiny_hypergraph, prng_key):
        conv = SparseUniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        sparse_hg = to_sparse(tiny_hypergraph)

        def loss_fn(model):
            return jnp.sum(model(sparse_hg))

        grads = jax.grad(loss_fn)(conv)
        assert not jnp.allclose(grads.linear.weight, 0.0)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:
    """All operations must be JIT-compatible."""

    def test_jit_sparse_v2e(self, tiny_hypergraph):
        sparse_hg = to_sparse(tiny_hypergraph)

        @jax.jit
        def run(x, indices, data):
            return sparse_vertex_to_edge(x, indices, data, sparse_hg.shape)

        result = run(
            tiny_hypergraph.node_features, sparse_hg.indices, sparse_hg.data
        )
        expected = tiny_hypergraph.incidence.T @ tiny_hypergraph.node_features
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_jit_sparse_e2v(self, tiny_hypergraph):
        sparse_hg = to_sparse(tiny_hypergraph)
        e = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        @jax.jit
        def run(e_feats, indices, data):
            return sparse_edge_to_vertex(e_feats, indices, data, sparse_hg.shape)

        result = run(e, sparse_hg.indices, sparse_hg.data)
        expected = tiny_hypergraph.incidence @ e
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_jit_sparse_conv(self, tiny_hypergraph, prng_key):
        conv = SparseUniGCNConv(in_dim=2, out_dim=4, key=prng_key)
        sparse_hg = to_sparse(tiny_hypergraph)
        jit_conv = eqx.filter_jit(conv)
        out_jit = jit_conv(sparse_hg)
        out_eager = conv(sparse_hg)
        assert jnp.allclose(out_jit, out_eager, atol=1e-6)

    def test_jit_to_dense(self, tiny_hypergraph):
        sparse_hg = to_sparse(tiny_hypergraph)

        @jax.jit
        def roundtrip(shg_nf, shg_idx, shg_data):
            shg = SparseHypergraph(
                node_features=shg_nf,
                indices=shg_idx,
                data=shg_data,
                shape=sparse_hg.shape,
            )
            dense = shg.to_dense()
            return dense.incidence

        result = roundtrip(
            sparse_hg.node_features, sparse_hg.indices, sparse_hg.data
        )
        assert jnp.allclose(result, tiny_hypergraph.incidence)


# ---------------------------------------------------------------------------
# Edge list construction
# ---------------------------------------------------------------------------


class TestFromEdgeListSparse:
    """Test building SparseHypergraph from edge lists."""

    def test_basic_construction(self):
        edges = [{0, 1, 2}, {1, 2, 3}]
        shg = from_edge_list_sparse(edges, num_nodes=4)
        assert shg.shape == (4, 2)
        assert shg.nnz == 6
        # Check that converting to dense matches from_edge_list
        dense_hg = hgx.from_edge_list([(0, 1, 2), (1, 2, 3)], num_nodes=4)
        recovered = shg.to_dense()
        assert jnp.allclose(recovered.incidence, dense_hg.incidence)

    def test_with_lists(self):
        edges = [[0, 1], [2, 3, 4]]
        shg = from_edge_list_sparse(edges, num_nodes=5)
        assert shg.shape == (5, 2)
        assert shg.nnz == 5

    def test_with_tuples(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        shg = from_edge_list_sparse(edges)
        assert shg.shape == (4, 3)
        assert shg.nnz == 6

    def test_inferred_num_nodes(self):
        edges = [{5, 10}]
        shg = from_edge_list_sparse(edges)
        assert shg.shape == (11, 1)

    def test_with_node_features(self):
        edges = [{0, 1}, {1, 2}]
        nf = jnp.eye(3)
        shg = from_edge_list_sparse(edges, node_features=nf)
        assert jnp.array_equal(shg.node_features, nf)

    def test_default_node_features(self):
        edges = [{0, 1}]
        shg = from_edge_list_sparse(edges, num_nodes=3)
        assert shg.node_features.shape == (3, 1)
        assert jnp.allclose(shg.node_features, jnp.ones((3, 1)))

    def test_matches_dense_from_edge_list(self):
        """Edge list sparse should produce equivalent structure to dense."""
        edges_sets = [{0, 1, 2}, {2, 3}, {0, 3, 4}]
        edges_tuples = [(0, 1, 2), (2, 3), (0, 3, 4)]
        nf = jnp.ones((5, 2))

        shg = from_edge_list_sparse(edges_sets, node_features=nf, num_nodes=5)
        dense_hg = hgx.from_edge_list(edges_tuples, num_nodes=5, node_features=nf)
        recovered = shg.to_dense()
        assert jnp.allclose(recovered.incidence, dense_hg.incidence)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestSparseHypergraphProperties:
    """Test properties of SparseHypergraph."""

    def test_num_nodes(self, tiny_hypergraph):
        shg = to_sparse(tiny_hypergraph)
        assert shg.num_nodes == 4

    def test_num_edges(self, tiny_hypergraph):
        shg = to_sparse(tiny_hypergraph)
        assert shg.num_edges == 2

    def test_nnz(self, tiny_hypergraph):
        shg = to_sparse(tiny_hypergraph)
        assert shg.nnz == 6

    def test_num_nodes_with_mask(self):
        shg = from_sparse_incidence(
            node_features=jnp.ones((5, 1)),
            indices=jnp.array([[0, 0], [1, 0]], dtype=jnp.int32),
            data=jnp.ones(2),
            shape=(5, 1),
            node_mask=jnp.array([True, True, True, False, False]),
        )
        assert shg.num_nodes == 3


# ---------------------------------------------------------------------------
# from_sparse_incidence constructor
# ---------------------------------------------------------------------------


class TestFromSparseIncidence:
    """Test the from_sparse_incidence constructor."""

    def test_basic(self):
        indices = jnp.array([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=jnp.int32)
        data = jnp.ones(4)
        nf = jnp.ones((3, 2))
        shg = from_sparse_incidence(nf, indices, data, (3, 2))
        assert shg.shape == (3, 2)
        assert shg.nnz == 4

    def test_with_geometry(self):
        indices = jnp.array([[0, 0]], dtype=jnp.int32)
        data = jnp.ones(1)
        nf = jnp.ones((2, 1))
        pos = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        shg = from_sparse_incidence(
            nf, indices, data, (2, 1), positions=pos, geometry="euclidean"
        )
        assert shg.geometry == "euclidean"
        assert jnp.array_equal(shg.positions, pos)


# ---------------------------------------------------------------------------
# Scale test
# ---------------------------------------------------------------------------


class TestScale:
    """Test that sparse representation handles larger hypergraphs."""

    def test_large_sparse(self, prng_key):
        """1000 nodes, 500 edges, ~5% density."""
        n, m = 1000, 500
        density = 0.05
        key1, key2 = jax.random.split(prng_key)

        # Generate random sparse incidence
        mask = jax.random.bernoulli(key1, p=density, shape=(n, m))
        H = mask.astype(jnp.float32)
        nf = jax.random.normal(key2, (n, 4))
        hg = hgx.from_incidence(H, node_features=nf)

        # Convert to sparse
        shg = to_sparse(hg)
        expected_nnz = int(jnp.sum(H != 0))
        assert shg.nnz == expected_nnz

        # Roundtrip
        recovered = shg.to_dense()
        assert jnp.allclose(recovered.incidence, H)

        # V->E and E->V match
        dense_e = H.T @ nf
        sparse_e = sparse_vertex_to_edge(nf, shg.indices, shg.data, shg.shape)
        assert jnp.allclose(sparse_e, dense_e, atol=1e-4)

        edge_feats = jnp.ones((m, 4))
        dense_v = H @ edge_feats
        sparse_v = sparse_edge_to_vertex(
            edge_feats, shg.indices, shg.data, shg.shape
        )
        assert jnp.allclose(sparse_v, dense_v, atol=1e-4)

    def test_large_conv(self, prng_key):
        """SparseUniGCNConv on a 1000x500 hypergraph."""
        n, m = 1000, 500
        density = 0.05
        k1, k2, k3 = jax.random.split(prng_key, 3)

        mask = jax.random.bernoulli(k1, p=density, shape=(n, m))
        H = mask.astype(jnp.float32)
        nf = jax.random.normal(k2, (n, 4))
        hg = hgx.from_incidence(H, node_features=nf)

        dense_conv = hgx.UniGCNConv(in_dim=4, out_dim=8, key=k3)
        sparse_conv = SparseUniGCNConv(in_dim=4, out_dim=8, key=k3)

        shg = to_sparse(hg)
        out_dense = dense_conv(hg)
        out_sparse = sparse_conv(shg)

        assert jnp.allclose(out_dense, out_sparse, atol=1e-4)
