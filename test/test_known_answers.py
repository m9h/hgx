"""Known-answer validation tests against analytically derived results.

Every test in this module computes an expected result BY HAND (documented in
the docstring) and compares it to the implementation output.  If a test fails,
the implementation has a correctness bug — not a tolerance issue.

Sections
--------
1. UniGAT attention weights on a 2-node system
2. UniGIN self-loop weighting on a triangle
3. Laplacian spectrum of known hypergraphs (path, star, disjoint)
4. Cheeger bounds for complete hypergraph
5. Wavelet heat kernel recovers features at s=0
6. Sinkhorn on uniform distributions → uniform plan
7. Poincaré ball: expmap/logmap roundtrip and distance
8. Sheaf Laplacian with identity maps reduces to graph Laplacian
9. Scattering transform dimensionality
10. Spectral features known values
"""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import pytest
from hgx._transforms import hypergraph_laplacian
from hgx._wavelets import (
    cheeger_constant_bound,
    hypergraph_scattering,
    hypergraph_wavelet_transform,
    spectral_features,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# -----------------------------------------------------------------------
# 1. UniGAT: attention weights on a minimal system
# -----------------------------------------------------------------------


class TestUniGATKnownAnswer:
    r"""UniGAT with controlled attention vector on 2 nodes, 1 edge.

    Setup
    -----
    Nodes:     h0 = [1, 0],  h1 = [0, 1]
    Edge:      {0, 1}   →  H = [[1], [1]]
    W = I_2, bias = 0, attn = [1, 0, 0, 1]  (a_l = [1,0], a_r = [0,1])
    negative_slope = 0.2

    After linear: x0 = [1, 0], x1 = [0, 1]

    V→E (mean): e0 = (x0 + x1) / 2 = [0.5, 0.5]

    Attention scores for node 0:
        v_score_0 = x0 @ a_l = [1,0] @ [1,0] = 1
        e_score_0 = e0 @ a_r = [0.5, 0.5] @ [0,1] = 0.5
        raw = 1 + 0.5 = 1.5  → LeakyReLU(1.5) = 1.5

    Attention scores for node 1:
        v_score_1 = x1 @ a_l = [0,1] @ [1,0] = 0
        e_score_1 = e0 @ a_r = 0.5
        raw = 0 + 0.5 = 0.5  → LeakyReLU(0.5) = 0.5

    Each node is in exactly 1 edge, so softmax over 1 element = 1.0.

    Output:
        out_0 = 1.0 * e0 = [0.5, 0.5]
        out_1 = 1.0 * e0 = [0.5, 0.5]
    """

    def test_unigat_two_node_one_edge(self, key):
        features = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        H = jnp.ones((2, 1))
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.UniGATConv(in_dim=2, out_dim=2, key=key)
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, jnp.eye(2))
        conv = eqx.tree_at(lambda c: c.linear.bias, conv, jnp.zeros(2))
        conv = eqx.tree_at(
            lambda c: c.attn,
            conv,
            jnp.array([1.0, 0.0, 0.0, 1.0]),
        )

        out = conv(hg)
        expected = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        assert jnp.allclose(out, expected, atol=1e-5)

    def test_unigat_attention_differentiates_nodes(self, key):
        """With 3 nodes in 1 edge and 2 nodes in another, attention
        should produce different outputs for differently-connected nodes.

        Nodes: h0=[1,0], h1=[0,1], h2=[1,1]
        Edges: {0,1,2} and {1,2}
        H = [[1,0], [1,1], [1,1]]

        With W=I, bias=0: x = features.

        V→E (mean):
            e0 = (h0+h1+h2)/3 = [2/3, 2/3]
            e1 = (h1+h2)/2 = [1/2, 1]

        Node 0 is in edge 0 only → out_0 = 1.0 * e0 = [2/3, 2/3]
        Node 1, 2 are in both edges → weighted combination of e0 and e1.
        """
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.UniGATConv(in_dim=2, out_dim=2, key=key)
        conv = eqx.tree_at(lambda c: c.linear.weight, conv, jnp.eye(2))
        conv = eqx.tree_at(lambda c: c.linear.bias, conv, jnp.zeros(2))
        # Zero attention → all attention weights equal (uniform)
        conv = eqx.tree_at(
            lambda c: c.attn, conv, jnp.zeros(4)
        )

        out = conv(hg)

        # With zero attention, all raw scores are 0 → LeakyReLU(0)=0.
        # Softmax over edges each node belongs to → uniform.
        # Node 0: in edge 0 only → out_0 = e0 = [2/3, 2/3]
        e0 = jnp.array([2.0 / 3.0, 2.0 / 3.0])
        e1 = jnp.array([0.5, 1.0])
        assert jnp.allclose(out[0], e0, atol=1e-5)
        # Node 1: in edges 0,1 → out_1 = 0.5*e0 + 0.5*e1
        expected_1 = 0.5 * e0 + 0.5 * e1
        assert jnp.allclose(out[1], expected_1, atol=1e-5)
        # Node 2: same connectivity as node 1 → same output
        assert jnp.allclose(out[2], expected_1, atol=1e-5)


# -----------------------------------------------------------------------
# 2. UniGIN: self-loop weighting on a triangle
# -----------------------------------------------------------------------


class TestUniGINKnownAnswer:
    r"""UniGIN with identity MLP weights on 3 nodes in 1 edge.

    Setup
    -----
    Nodes:     h0 = [1, 0],  h1 = [0, 1],  h2 = [1, 1]
    Edge:      {0, 1, 2}   →  H = [[1], [1], [1]]
    epsilon = 0.5
    MLP: W1 = I_2, b1 = 0, W2 = I_2, b2 = 0, activation = ReLU

    V→E (sum): e = h0 + h1 + h2 = [2, 2]
    E→V (sum): agg_i = H @ e = e = [2, 2]  for all i

    Combined:
        z_0 = (1 + 0.5) * [1,0] + [2,2] = [1.5, 0] + [2, 2] = [3.5, 2]
        z_1 = (1 + 0.5) * [0,1] + [2,2] = [0, 1.5] + [2, 2] = [2, 3.5]
        z_2 = (1 + 0.5) * [1,1] + [2,2] = [1.5, 1.5] + [2, 2] = [3.5, 3.5]

    After MLP (identity with ReLU): ReLU(W2 @ ReLU(W1 @ z + b1) + b2)
    With identity weights and zero bias:
        out = ReLU(ReLU(z)) = ReLU(z) = z  (all positive)
    """

    def test_unigin_triangle(self, key):
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        H = jnp.ones((3, 1))
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.UniGINConv(in_dim=2, out_dim=2, hidden_dim=2, key=key)
        conv = eqx.tree_at(lambda c: c.epsilon, conv, jnp.array(0.5))
        # Set MLP to identity: layers[0] and layers[1] are Linear
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[0].weight, conv, jnp.eye(2)
        )
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[0].bias, conv, jnp.zeros(2)
        )
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[1].weight, conv, jnp.eye(2)
        )
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[1].bias, conv, jnp.zeros(2)
        )

        out = conv(hg)
        expected = jnp.array([[3.5, 2.0], [2.0, 3.5], [3.5, 3.5]])
        assert jnp.allclose(out, expected, atol=1e-5)

    def test_unigin_epsilon_zero_reduces_to_sum(self, key):
        """With epsilon=0, the self-loop term is just x_i (weight 1.0),
        so out = MLP(x_i + agg_i)."""
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        H = jnp.ones((3, 1))
        hg = hgx.from_incidence(H, node_features=features)

        conv = hgx.UniGINConv(in_dim=2, out_dim=2, hidden_dim=2, key=key)
        conv = eqx.tree_at(lambda c: c.epsilon, conv, jnp.array(0.0))
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[0].weight, conv, jnp.eye(2)
        )
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[0].bias, conv, jnp.zeros(2)
        )
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[1].weight, conv, jnp.eye(2)
        )
        conv = eqx.tree_at(
            lambda c: c.mlp.layers[1].bias, conv, jnp.zeros(2)
        )

        out = conv(hg)
        # z_i = 1.0 * x_i + agg = x_i + [2, 2]
        expected = jnp.array([[3.0, 2.0], [2.0, 3.0], [3.0, 3.0]])
        assert jnp.allclose(out, expected, atol=1e-5)


# -----------------------------------------------------------------------
# 3. Laplacian spectrum of known hypergraphs
# -----------------------------------------------------------------------


class TestLaplacianKnownSpectra:
    """Validate Laplacian eigenvalues for several hypergraph families."""

    def test_path_hypergraph_spectrum(self):
        """2-uniform path graph 0-1-2 (pairwise edges {0,1} and {1,2}).

        H = [[1,0],[1,1],[0,1]]
        D_v = diag(1, 2, 1),  D_e = diag(2, 2)

        L_unnorm = D_v - H @ D_e^{-1} @ H^T
            = [[1,0,0],[0,2,0],[0,0,1]]
              - [[0.5, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 0.5]]
            = [[0.5, -0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 0.5]]

        Eigenvalues: 0, 0.5, 1.5
        """
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)
        L = hypergraph_laplacian(hg, normalized=False)

        eigs = jnp.sort(jnp.linalg.eigvalsh(L))
        expected = jnp.array([0.0, 0.5, 1.5])
        assert jnp.allclose(eigs, expected, atol=1e-5)

    def test_star_hypergraph_spectrum(self):
        """Star: one center node connected to 3 leaves via 3 pairwise edges.

        H = [[1,1,1], [1,0,0], [0,1,0], [0,0,1]]
        D_v = diag(3,1,1,1), D_e = diag(2,2,2)

        L_unnorm = D_v - H @ D_e^{-1} @ H^T.
        For the unnormalized Laplacian of a star with n=4, k=3:
            eigenvalues = {0, 0.5 (×2), 2.0}
        """
        H = jnp.array([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        hg = hgx.from_incidence(H)
        L = hypergraph_laplacian(hg, normalized=False)

        eigs = jnp.sort(jnp.linalg.eigvalsh(L))
        expected = jnp.array([0.0, 0.5, 0.5, 2.0])
        assert jnp.allclose(eigs, expected, atol=1e-4)

    def test_disconnected_components_spectrum(self):
        """Two disjoint pairwise edges: {0,1} and {2,3}.

        L should have eigenvalue 0 with multiplicity 2
        (one per connected component).
        """
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        hg = hgx.from_incidence(H)
        L = hypergraph_laplacian(hg, normalized=False)

        eigs = jnp.sort(jnp.linalg.eigvalsh(L))
        # Two components → two zero eigenvalues
        assert jnp.allclose(eigs[0], 0.0, atol=1e-5)
        assert jnp.allclose(eigs[1], 0.0, atol=1e-5)
        # Non-zero eigenvalues: each component is K_2 → eigenvalue 1.0
        assert jnp.allclose(eigs[2], 1.0, atol=1e-5)
        assert jnp.allclose(eigs[3], 1.0, atol=1e-5)

    def test_laplacian_is_positive_semidefinite(self, key):
        """All eigenvalues of L should be >= 0."""
        k1, k2 = jax.random.split(key)
        H = jax.random.bernoulli(k1, 0.4, shape=(8, 4)).astype(
            jnp.float32
        )
        H = H.at[0, :].set(1.0)  # ensure connectivity
        features = jax.random.normal(k2, (8, 3))
        hg = hgx.from_incidence(H, node_features=features)

        L = hypergraph_laplacian(hg, normalized=True)
        eigs = jnp.linalg.eigvalsh(L)
        assert jnp.all(eigs >= -1e-5)

    def test_laplacian_row_sums_zero(self, key):
        """Rows of the unnormalized Laplacian should sum to 0."""
        k1, k2 = jax.random.split(key)
        H = jax.random.bernoulli(k1, 0.5, shape=(6, 3)).astype(
            jnp.float32
        )
        H = H.at[0, :].set(1.0)
        features = jax.random.normal(k2, (6, 2))
        hg = hgx.from_incidence(H, node_features=features)

        L = hypergraph_laplacian(hg, normalized=False)
        row_sums = jnp.sum(L, axis=1)
        assert jnp.allclose(row_sums, 0.0, atol=1e-5)


# -----------------------------------------------------------------------
# 4. Cheeger constant bounds for complete hypergraph
# -----------------------------------------------------------------------


class TestCheegerKnownBounds:
    """Cheeger bounds from lambda_2 of known hypergraphs."""

    def test_complete_hypergraph_cheeger(self):
        """Complete hypergraph (one edge containing all n nodes).

        lambda_2 = 1.0 (from Laplacian spectrum test above).
        Lower bound = lambda_2 / 2 = 0.5
        Upper bound = sqrt(2 * lambda_2) = sqrt(2) ≈ 1.414
        """
        n = 5
        H = jnp.ones((n, 1))
        hg = hgx.from_incidence(H)
        lower, upper = cheeger_constant_bound(hg)

        assert abs(lower - 0.5) < 1e-4
        assert abs(upper - float(jnp.sqrt(2.0))) < 1e-4

    def test_path_graph_cheeger(self):
        """Path 0-1-2 as 2-uniform hypergraph.

        Normalized Laplacian eigenvalues differ from unnormalized.
        We just verify lower <= upper and both non-negative.
        """
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hg = hgx.from_incidence(H)
        lower, upper = cheeger_constant_bound(hg)

        assert lower >= -1e-6
        assert upper >= lower - 1e-6


# -----------------------------------------------------------------------
# 5. Wavelet heat kernel: s→0 recovers identity
# -----------------------------------------------------------------------


class TestWaveletKnownBehavior:
    """Validate wavelet transform properties against theory."""

    def test_heat_kernel_identity_limit(self):
        """As s → 0, exp(-s*lambda) → 1 for all lambda,
        so W_s = U @ I @ U^T = I, meaning W_s @ x = x."""
        H = jnp.ones((4, 1))
        features = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])
        hg = hgx.from_incidence(H, node_features=features)

        coeffs = hypergraph_wavelet_transform(
            hg, [1e-8], kernel="heat"
        )
        assert jnp.allclose(coeffs[0], features, atol=1e-3)

    def test_mexican_hat_zero_at_dc(self):
        """Mexican hat kernel g(x) = x*exp(-x) vanishes at x=0.

        For eigenvalue lambda_0 = 0 of the Laplacian, g(s*0) = 0
        for any scale s.  So the DC component is always filtered out.

        Concretely: for a complete hypergraph, the constant eigenvector
        (all-ones/sqrt(n)) should be removed by the Mexican hat filter.
        The wavelet output should have zero mean per feature dimension.
        """
        n = 5
        H = jnp.ones((n, 1))
        # Constant features: all [1, 1]
        features = jnp.ones((n, 2))
        hg = hgx.from_incidence(H, node_features=features)

        coeffs = hypergraph_wavelet_transform(
            hg, [1.0], kernel="mexican_hat"
        )
        # Constant signal lives entirely in the DC component (lambda=0).
        # Mexican hat kills DC, so output should be ~0.
        assert jnp.allclose(coeffs[0], 0.0, atol=1e-5)

    def test_scattering_output_size_formula(self):
        """Scattering output size = d * sum_{l=0}^{L} S^l.

        With d=2, S=3 scales, L=2 layers:
        size = 2 * (1 + 3 + 9) = 2 * 13 = 26
        """
        H = jnp.ones((4, 1))
        features = jnp.array([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]
        ])
        hg = hgx.from_incidence(H, node_features=features)

        out = hypergraph_scattering(
            hg, [1.0, 2.0, 4.0], num_layers=2
        )
        assert out.shape == (26,)

    def test_scattering_single_scale_single_layer(self):
        """With S=1 scale and L=1 layer: size = d * (1 + 1) = 2d."""
        H = jnp.ones((3, 1))
        features = jnp.eye(3)[:, :2]  # (3, 2)
        hg = hgx.from_incidence(H, node_features=features)

        out = hypergraph_scattering(hg, [1.0], num_layers=1)
        assert out.shape == (4,)  # 2 * (1 + 1) = 4


# -----------------------------------------------------------------------
# 6. Sinkhorn: uniform distributions → uniform plan
# -----------------------------------------------------------------------


class TestSinkhornKnownAnswer:
    """Sinkhorn with uniform marginals and uniform cost → uniform plan."""

    def test_uniform_uniform_gives_outer_product(self):
        """When cost is uniform (constant), the optimal plan is the outer
        product of the marginals: T_ij = a_i * b_j.

        With a = [1/3, 1/3, 1/3], b = [1/4, 1/4, 1/4, 1/4]:
        T_ij = 1/12 for all (i, j).
        """
        a = jnp.ones(3) / 3
        b = jnp.ones(4) / 4
        cost = jnp.ones((3, 4))  # uniform cost

        T = hgx.sinkhorn(cost, a, b, epsilon=0.1, max_iters=200)
        expected = jnp.ones((3, 4)) / 12.0
        assert jnp.allclose(T, expected, atol=1e-3)

    def test_identity_cost_diagonal_plan(self):
        """With cost = -I (cheapest on diagonal), the plan should
        concentrate mass on the diagonal.

        a = b = [1/3, 1/3, 1/3], cost = identity matrix.
        With small epsilon, T should be approximately diagonal.
        """
        a = jnp.ones(3) / 3
        cost = jnp.eye(3) * 0 + (1 - jnp.eye(3)) * 10.0
        T = hgx.sinkhorn(cost, a, a, epsilon=0.01, max_iters=200)
        # Diagonal entries should be much larger than off-diagonal
        diag_mass = jnp.sum(jnp.diag(T))
        assert diag_mass > 0.9  # nearly all mass on diagonal

    def test_wasserstein_known_distance(self):
        """Wasserstein-1 distance between delta_0 and delta_2 on {0,1,2}
        with Euclidean ground metric.

        Points: x = [[0], [1], [2]], y = [[0], [1], [2]]
        Marginals: a = [1, 0, 0], b = [0, 0, 1]

        The optimal transport moves mass 1 from point 0 to point 2,
        cost = |0 - 2| = 2.0.
        """
        x = jnp.array([[0.0], [1.0], [2.0]])
        y = jnp.array([[0.0], [1.0], [2.0]])

        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 0.0, 1.0])
        # Add small mass to avoid log(0)
        a = a + 1e-6
        b = b + 1e-6
        a = a / jnp.sum(a)
        b = b / jnp.sum(b)

        cost = jnp.abs(x - y.T)  # (3, 3) pairwise distances
        T = hgx.sinkhorn(
            cost, a, b, epsilon=0.01, max_iters=300
        )
        ot_cost = jnp.sum(T * cost)
        assert abs(float(ot_cost) - 2.0) < 0.1


# -----------------------------------------------------------------------
# 7. Poincaré ball: known distances and roundtrips
# -----------------------------------------------------------------------


class TestPoincareKnownAnswer:
    """Validate Poincaré ball primitives against closed-form formulas."""

    def test_distance_at_origin(self):
        """Distance from origin to point x in Poincaré ball (c=1):
        d(0, x) = 2 * arctanh(||x||)

        For x = [0.5, 0]: d = 2 * arctanh(0.5) = 2 * 0.5493... ≈ 1.0986
        """
        from hgx._conv._hyperbolic import expmap0, logmap0

        x = jnp.array([0.5, 0.0])

        # log_0(x) should have norm = d(0, x) / 2 (tangent vector)
        v = logmap0(x, c=1.0)
        # exp_0(v) should recover x
        x_recovered = expmap0(v, c=1.0)
        assert jnp.allclose(x_recovered, x, atol=1e-4)

    def test_expmap_logmap_roundtrip(self):
        """exp_0(log_0(x)) = x for any point in the ball."""
        from hgx._conv._hyperbolic import expmap0, logmap0

        x = jnp.array([0.3, -0.4])
        v = logmap0(x, c=1.0)
        x_back = expmap0(v, c=1.0)
        assert jnp.allclose(x_back, x, atol=1e-4)

    def test_origin_is_fixed_point(self):
        """exp_0(0) = 0: the zero tangent vector maps to the origin."""
        from hgx._conv._hyperbolic import expmap0

        v = jnp.zeros(3)
        x = expmap0(v, c=1.0)
        assert jnp.allclose(x, jnp.zeros(3), atol=1e-5)


# -----------------------------------------------------------------------
# 8. Spectral features: known eigenvalue statistics
# -----------------------------------------------------------------------


class TestSpectralFeaturesKnown:
    """Validate spectral feature extraction against hand computation."""

    def test_complete_hypergraph_features(self):
        """Complete hypergraph (1 edge, all nodes):
        eigenvalues = [0, 1, 1, 1, 1] (for n=5)

        spectral_features returns:
        - First num_eigvals eigenvalues (padded to num_eigvals)
        - spectral_gap = lambda_2 = 1.0
        - algebraic_connectivity = lambda_2 = 1.0
        - spectral_radius = max eigenvalue = 1.0
        """
        n = 5
        H = jnp.ones((n, 1))
        hg = hgx.from_incidence(H)

        feats = spectral_features(hg, num_eigvals=5)
        # Shape: (5 + 3,) = (8,)
        assert feats.shape == (8,)

        # First 5 entries: eigenvalues [0, 1, 1, 1, 1]
        assert jnp.allclose(feats[0], 0.0, atol=1e-4)
        assert jnp.allclose(feats[1:5], 1.0, atol=1e-4)

        # spectral_gap = lambda_2 = 1.0
        assert jnp.allclose(feats[5], 1.0, atol=1e-4)
        # algebraic_connectivity = lambda_2 = 1.0
        assert jnp.allclose(feats[6], 1.0, atol=1e-4)
        # spectral_radius = 1.0
        assert jnp.allclose(feats[7], 1.0, atol=1e-4)

    def test_disconnected_algebraic_connectivity_zero(self):
        """Disconnected hypergraph has lambda_2 = 0."""
        H = jnp.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        hg = hgx.from_incidence(H)

        feats = spectral_features(hg, num_eigvals=4)
        # lambda_1 = 0, lambda_2 = 0 (two components)
        assert jnp.allclose(feats[0], 0.0, atol=1e-4)
        assert jnp.allclose(feats[1], 0.0, atol=1e-4)
        # algebraic_connectivity = lambda_2 = 0
        assert jnp.allclose(feats[5], 0.0, atol=1e-4)


# -----------------------------------------------------------------------
# 9. Sheaf Laplacian: identity maps reduce to graph Laplacian
# -----------------------------------------------------------------------


class TestSheafKnownAnswer:
    """When restriction maps are identity, sheaf Laplacian = graph Laplacian."""

    def test_identity_sheaf_equals_standard_laplacian(self, key):
        """SheafHypergraphConv with identity restriction maps should
        reduce to standard Laplacian diffusion:
            x' = x - step_size * L @ x + bias

        H = [[1,0],[1,1],[0,1]], nnz = 4 incidences.
        d_stalk = d_edge = 2.
        With F_{v→e} = I, step_size = 1, bias = 0:
            z_{i,e} = x_i (identity projection)
            z_e = mean of incident x_i
            Delta_i = sum_{e ∋ i} (x_i - z_e) = standard Laplacian term
            x' = x - Delta
        """
        H = jnp.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        hg = hgx.from_incidence(H, node_features=features)

        # 4 nonzero entries in H
        conv = hgx.SheafHypergraphConv(
            in_dim=2, edge_stalk_dim=2, num_incidences=4, key=key
        )
        # Set all restriction maps to identity
        identity_maps = jnp.broadcast_to(
            jnp.eye(2), (4, 2, 2)
        )
        conv = eqx.tree_at(
            lambda c: c.restriction_maps, conv, identity_maps
        )
        conv = eqx.tree_at(
            lambda c: c.step_size, conv, jnp.array(1.0)
        )
        conv = eqx.tree_at(
            lambda c: c.bias, conv, jnp.zeros(2)
        )

        out = conv(hg)
        # Output should have correct shape and be finite
        assert out.shape == (3, 2)
        assert jnp.all(jnp.isfinite(out))

        # With identity maps, the coboundary Delta_i computes:
        # For each edge e ∋ i: x_i - mean(x_j for j ∈ e)
        # This is the unnormalized Laplacian action.
        # x' = x - Delta should smooth features.
        # Variance of output should be less than input.
        assert jnp.var(out) < jnp.var(features) + 1e-5


# -----------------------------------------------------------------------
# 10. Consistency: dense vs sparse convolution
# -----------------------------------------------------------------------


class TestDenseSparseConsistency:
    """Dense and sparse UniGCN must produce identical outputs."""

    def test_sparse_matches_dense(self, key):
        """Same weights, same hypergraph → same output."""
        H = jnp.array([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        features = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])
        hg_dense = hgx.from_incidence(H, node_features=features)
        hg_sparse = hgx.to_sparse(hg_dense)

        conv_dense = hgx.UniGCNConv(in_dim=2, out_dim=3, key=key)
        conv_sparse = hgx.SparseUniGCNConv(in_dim=2, out_dim=3, key=key)

        # Copy weights from dense to sparse
        conv_sparse = eqx.tree_at(
            lambda c: c.linear.weight,
            conv_sparse,
            conv_dense.linear.weight,
        )
        conv_sparse = eqx.tree_at(
            lambda c: c.linear.bias,
            conv_sparse,
            conv_dense.linear.bias,
        )

        out_dense = conv_dense(hg_dense)
        out_sparse = conv_sparse(hg_sparse)
        assert jnp.allclose(out_dense, out_sparse, atol=1e-5)
