"""Hand-computed mathematical validation tests.

Each test constructs a small system with known weight matrices (set via
eqx.tree_at) and compares the implementation output against a manually
worked expected result.  All assertions use jnp.allclose with atol=1e-5.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import hgx


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# -----------------------------------------------------------------------
# 1. UniGCN on a 3-node, 1-hyperedge system
# -----------------------------------------------------------------------

class TestUniGCNThreeNodeOneEdge:
    """UniGCN with identity weights on a triangle hyperedge.

    Setup
    -----
    Nodes:      h0 = [1, 0],  h1 = [0, 1],  h2 = [1, 1]
    Hyperedge:  {0, 1, 2}   →  H = [[1], [1], [1]]
    Weights:    W = I_2, bias = 0

    Hand computation
    ----------------
    x = W @ h = h   (identity)
    d_v = [1, 1, 1]  (each node in 1 edge)
    d_e = [3]         (3 nodes in the edge)

    Stage 1 — vertex→hyperedge:
        e = (1/3)(x0 + x1 + x2) = (1/3)([1,0]+[0,1]+[1,1]) = [2/3, 2/3]

    Stage 2 — hyperedge→vertex:
        out_i = (1/d_v[i]) * H[i,:] @ e = (1/1) * e = [2/3, 2/3]   ∀ i
    """

    def test_unigcn_three_node_one_edge(self, key):
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        H = jnp.ones((3, 1))
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.UniGCNConv(in_dim=2, out_dim=2, key=key)
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, jnp.eye(2))
        conv = eqx.tree_at(lambda c: c.linear.bias, conv, jnp.zeros(2))

        out = conv(hg)
        expected = jnp.full((3, 2), 2.0 / 3.0)
        assert jnp.allclose(out, expected, atol=1e-5)


# -----------------------------------------------------------------------
# 2. UniGCN reduces to GCN-with-self-loops on a pairwise graph
# -----------------------------------------------------------------------

class TestUniGCNPairwiseReduction:
    """UniGCN on a 2-uniform hypergraph (path graph 0–1–2).

    Setup
    -----
    Hyperedges: {0,1} and {1,2}  →  H = [[1,0],[1,1],[0,1]]
    Features:   h0 = [1,0],  h1 = [0,1],  h2 = [1,1]
    Weights:    W = I_2, bias = 0

    Hand computation
    ----------------
    d_v = [1, 2, 1],   d_e = [2, 2]

    Stage 1:
        e0 = (1/2)(h0 + h1) = [1/2, 1/2]
        e1 = (1/2)(h1 + h2) = [1/2, 1]

    Stage 2:
        (H @ e)[0] = e0            = [1/2, 1/2]
        (H @ e)[1] = e0 + e1       = [1, 3/2]
        (H @ e)[2] = e1            = [1/2, 1]

        out[0] = (1/1) [1/2, 1/2]  = [0.5, 0.5]
        out[1] = (1/2) [1, 3/2]    = [0.5, 0.75]
        out[2] = (1/1) [1/2, 1]    = [0.5, 1.0]

    Equivalently, for any 2-uniform hypergraph UniGCN computes:
        out = (D⁻¹AX + X) / 2
    i.e. the GCN formula with renormalized self-loops.
    """

    def test_unigcn_path_graph(self, key):
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.UniGCNConv(in_dim=2, out_dim=2, key=key)
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, jnp.eye(2))
        conv = eqx.tree_at(lambda c: c.linear.bias, conv, jnp.zeros(2))

        out = conv(hg)
        expected = jnp.array([[0.5, 0.5], [0.5, 0.75], [0.5, 1.0]])
        assert jnp.allclose(out, expected, atol=1e-5)

    def test_equals_gcn_with_self_loops(self, key):
        """Verify (D⁻¹AX + X)/2 matches UniGCN on any 2-uniform hypergraph."""
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.UniGCNConv(in_dim=2, out_dim=2, key=key)
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, jnp.eye(2))
        conv = eqx.tree_at(lambda c: c.linear.bias, conv, jnp.zeros(2))

        out = conv(hg)

        # Compute (D⁻¹AX + X)/2 directly
        A = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
        D_inv = jnp.diag(jnp.array([1.0, 0.5, 1.0]))
        gcn_with_self_loops = (D_inv @ A @ features + features) / 2.0

        assert jnp.allclose(out, gcn_with_self_loops, atol=1e-5)


# -----------------------------------------------------------------------
# 3. THNN on a 3-node, 1-hyperedge system
# -----------------------------------------------------------------------

class TestTHNNThreeNodeOneEdge:
    """THNN with controlled weights on a 3-node triangle hyperedge.

    Setup
    -----
    Nodes:       h0 = [1,0,0],  h1 = [0,1,0],  h2 = [0,0,1]  (one-hot)
    Hyperedge:   {0, 1, 2}   →   H = [[1], [1], [1]]
    in_dim = 3, rank = 3, out_dim = 3
    Theta weight (3×4):  [[1,0,0,1], [0,1,0,1], [0,0,1,1]]
        → projects [h; 1] to h + [1,1,1]  (identity + constant offset)
    Q weight = I_3, Q bias = [0, 0, 0]

    Hand computation
    ----------------
    Augmented:  [h0; 1] = [1,0,0,1],  [h1; 1] = [0,1,0,1],  [h2; 1] = [0,0,1,1]

    After theta projection (no bias):
        z0 = Θ @ [1,0,0,1] = [1+1, 0+1, 0+1] = [2, 1, 1]
        z1 = Θ @ [0,1,0,1] = [0+1, 1+1, 0+1] = [1, 2, 1]
        z2 = Θ @ [0,0,1,1] = [0+1, 0+1, 1+1] = [1, 1, 2]

    Element-wise product over hyperedge:
        m = z0 ⊙ z1 ⊙ z2 = [2·1·1, 1·2·1, 1·1·2] = [2, 2, 2]

    After tanh:
        tanh(m) = [tanh(2), tanh(2), tanh(2)]

    After Q (identity, zero bias):
        e_out = tanh(m)

    Degree normalization (d_v = [1,1,1]):
        out_i = (1/1) · H[i,:] @ e_out = [tanh(2), tanh(2), tanh(2)]   ∀ i
    """

    def test_thnn_three_node_product(self, key):
        features = jnp.eye(3)  # one-hot: [1,0,0], [0,1,0], [0,0,1]
        H = jnp.ones((3, 1))
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.THNNConv(in_dim=3, out_dim=3, rank=3, key=key)

        # Theta: identity-like, maps [h; 1] → h + [1,1,1]
        theta_w = jnp.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ])
        conv = eqx.tree_at(lambda c: c.theta.weight, conv, theta_w)

        # Q: identity with zero bias
        conv = eqx.tree_at(lambda c: c.q.weight, conv, jnp.eye(3))
        conv = eqx.tree_at(lambda c: c.q.bias, conv, jnp.zeros(3))

        out = conv(hg)

        t2 = jnp.tanh(jnp.array(2.0))
        expected = jnp.full((3, 3), t2)
        assert jnp.allclose(out, expected, atol=1e-5)

    def test_thnn_product_versus_direct(self, key):
        """Verify the log-domain product aggregation matches direct computation.

        Uses all-positive features [0.5, 0.3], [0.2, 0.4], [0.7, 0.1]
        with in_dim=2, rank=3, identity-like theta, identity Q.

        z0 = [0.5+1, 0.3+1, 1] = [1.5, 1.3, 1]   (theta adds 1 from constant col)
        z1 = [0.2+1, 0.4+1, 1] = [1.2, 1.4, 1]
        z2 = [0.7+1, 0.1+1, 1] = [1.7, 1.1, 1]

        m = [1.5·1.2·1.7, 1.3·1.4·1.1, 1·1·1]
          = [3.06, 2.002, 1.0]

        out_i = Q @ tanh(m) = tanh(m) for all i   (d_v = [1,1,1])
        """
        features = jnp.array([[0.5, 0.3], [0.2, 0.4], [0.7, 0.1]])
        H = jnp.ones((3, 1))
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.THNNConv(in_dim=2, out_dim=3, rank=3, key=key)

        # Theta (3×3): identity-like + constant column
        theta_w = jnp.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        conv = eqx.tree_at(lambda c: c.theta.weight, conv, theta_w)
        conv = eqx.tree_at(lambda c: c.q.weight, conv, jnp.eye(3))
        conv = eqx.tree_at(lambda c: c.q.bias, conv, jnp.zeros(3))

        out = conv(hg)

        m = jnp.array([1.5 * 1.2 * 1.7, 1.3 * 1.4 * 1.1, 1.0 * 1.0 * 1.0])
        expected = jnp.broadcast_to(jnp.tanh(m), (3, 3))
        assert jnp.allclose(out, expected, atol=1e-5)


# -----------------------------------------------------------------------
# 4. Clique expansion correctness
# -----------------------------------------------------------------------

class TestCliqueExpansion:
    """Verify clique expansion on small hand-worked examples."""

    def test_single_triangle_hyperedge(self):
        """Hyperedge {0,1,2} → complete graph K₃.

        H = [[1],[1],[1]]
        HH^T = [[1,1,1],[1,1,1],[1,1,1]]
        A = HH^T - diag = all-ones minus diagonal → K₃ adjacency
        """
        H = jnp.ones((3, 1))
        hg = hgx.from_incidence(H)
        A = hgx.clique_expansion(hg)
        expected = jnp.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        assert jnp.allclose(A, expected, atol=1e-5)

    def test_two_overlapping_hyperedges(self):
        """Hyperedges {0,1} and {1,2} → path graph 0–1–2.

        H = [[1,0],[1,1],[0,1]]
        HH^T = [[1,1,0],[1,2,1],[0,1,1]]
        After removing diagonal and binarizing: path adjacency.
        """
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)
        A = hgx.clique_expansion(hg)
        expected = jnp.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        assert jnp.allclose(A, expected, atol=1e-5)

    def test_disjoint_hyperedges(self):
        """Disjoint hyperedges {0,1} and {2,3} → two disconnected edges.

        No shared nodes, so the clique expansion has no cross-component edges.
        """
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        hg = hgx.from_incidence(H)
        A = hgx.clique_expansion(hg)
        expected = jnp.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        assert jnp.allclose(A, expected, atol=1e-5)


# -----------------------------------------------------------------------
# 5. Laplacian eigenvalue test
# -----------------------------------------------------------------------

class TestLaplacianEigenvalues:
    """Eigenvalue spectrum of the normalized hypergraph Laplacian.

    Complete hypergraph: one hyperedge containing all n nodes.
        H = 1_n,   d_v = [1,...,1],   d_e = [n]

        L_sym = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
              = I - (1/n) J       (J = all-ones matrix)

    Eigenvalues of (1/n)J:  1 (multiplicity 1), 0 (multiplicity n-1)
    Eigenvalues of L_sym:   0 (multiplicity 1), 1 (multiplicity n-1)

    The constant vector [1,...,1]/√n is the eigenvector for eigenvalue 0.
    """

    @pytest.mark.parametrize("n", [4, 5, 10])
    def test_single_all_inclusive_hyperedge(self, n):
        H = jnp.ones((n, 1))
        hg = hgx.from_incidence(H)
        L = hgx.hypergraph_laplacian(hg, normalized=True)

        eigenvalues = jnp.linalg.eigvalsh(L)
        eigenvalues = jnp.sort(eigenvalues)

        # Smallest eigenvalue should be 0 (multiplicity 1)
        assert jnp.allclose(eigenvalues[0], 0.0, atol=1e-5)

        # All other eigenvalues should be 1
        assert jnp.allclose(eigenvalues[1:], 1.0, atol=1e-5)

    def test_eigenvalue_zero_eigenvector(self):
        """The constant vector is the null eigenvector of L_sym."""
        n = 5
        H = jnp.ones((n, 1))
        hg = hgx.from_incidence(H)
        L = hgx.hypergraph_laplacian(hg, normalized=True)

        v_const = jnp.ones(n) / jnp.sqrt(n)
        Lv = L @ v_const
        assert jnp.allclose(Lv, jnp.zeros(n), atol=1e-5)

    def test_unnormalized_laplacian(self):
        """Unnormalized L = D_v - H D_e^{-1} H^T = I - (1/n)J
        for a single all-inclusive edge (d_v = 1 for all nodes).
        Same eigenvalues: 0 (×1), 1 (×n-1).
        """
        n = 6
        H = jnp.ones((n, 1))
        hg = hgx.from_incidence(H)
        L = hgx.hypergraph_laplacian(hg, normalized=False)

        eigenvalues = jnp.sort(jnp.linalg.eigvalsh(L))
        assert jnp.allclose(eigenvalues[0], 0.0, atol=1e-5)
        assert jnp.allclose(eigenvalues[1:], 1.0, atol=1e-5)
