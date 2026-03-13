"""Tests for topological feature extraction."""

import hgx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hgx._topology import (
    hodge_laplacians,
    persistence_image,
    persistence_landscape,
    TopologicalLayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _triangle_hypergraph():
    """A single triangle hyperedge {0, 1, 2}.

    Clique expansion gives K3.  The flag complex fills the 2-simplex,
    so beta_0=1, beta_1=0, beta_2=0.
    """
    return hgx.from_edge_list([(0, 1, 2)])


def _path_hypergraph():
    """Path graph 0-1-2-3 as pairwise hyperedges.

    beta_0 = 1 (connected), beta_1 = 0 (tree-like).
    """
    return hgx.from_edge_list([(0, 1), (1, 2), (2, 3)])


def _two_components():
    """Two disconnected edges: {0,1} and {2,3}.

    beta_0 = 2.
    """
    return hgx.from_edge_list([(0, 1), (2, 3)])


# ---------------------------------------------------------------------------
# persistence_landscape
# ---------------------------------------------------------------------------


class TestPersistenceLandscape:
    """Test persistence landscape vectorisation."""

    def test_shape(self):
        diagram = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        result = persistence_landscape(diagram, num_landscapes=5, resolution=100)
        assert result.shape == (5, 100)

    def test_more_landscapes_than_pairs(self):
        diagram = np.array([[0.0, 1.0]])
        result = persistence_landscape(diagram, num_landscapes=5, resolution=50)
        assert result.shape == (5, 50)
        # Landscapes beyond the first should be all zeros.
        np.testing.assert_allclose(result[1:], 0.0)

    def test_empty_diagram(self):
        diagram = np.zeros((0, 2))
        result = persistence_landscape(diagram, num_landscapes=3, resolution=20)
        assert result.shape == (3, 20)
        np.testing.assert_allclose(result, 0.0)

    def test_nonnegative(self):
        diagram = np.array([[0.0, 1.0], [0.5, 3.0]])
        result = persistence_landscape(diagram)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# persistence_image
# ---------------------------------------------------------------------------


class TestPersistenceImage:
    """Test persistence image vectorisation."""

    def test_shape(self):
        diagram = np.array([[0.0, 1.0], [0.5, 2.0]])
        result = persistence_image(diagram, resolution=50, sigma=0.1)
        assert result.shape == (50, 50)

    def test_empty_diagram(self):
        diagram = np.zeros((0, 2))
        result = persistence_image(diagram, resolution=30)
        assert result.shape == (30, 30)
        np.testing.assert_allclose(result, 0.0)

    def test_nonnegative(self):
        diagram = np.array([[0.0, 1.0], [0.5, 2.0]])
        result = persistence_image(diagram, resolution=25, sigma=0.5)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# TopologicalLayer (differentiable, JIT-compatible)
# ---------------------------------------------------------------------------


class TestTopologicalLayer:
    """Test the differentiable topological layer."""

    def test_output_shape(self):
        layer = TopologicalLayer(num_eigvals=8, bandwidth=1.0)
        features = jnp.ones((5, 3))
        out = layer(features)
        assert out.shape == (16,)  # 2 * num_eigvals

    def test_small_input(self):
        """When n < num_eigvals, output is still correctly shaped."""
        layer = TopologicalLayer(num_eigvals=8, bandwidth=1.0)
        features = jnp.ones((3, 2))
        out = layer(features)
        assert out.shape == (16,)

    def test_jit_compatible(self):
        layer = TopologicalLayer(num_eigvals=4, bandwidth=1.0)
        features = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        jitted = jax.jit(layer)
        out = jitted(features)
        assert out.shape == (8,)
        # Should agree with non-jitted version.
        np.testing.assert_allclose(out, layer(features), atol=1e-5)

    def test_differentiable(self):
        layer = TopologicalLayer(num_eigvals=4, bandwidth=1.0)
        features = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        def loss_fn(feats):
            return jnp.sum(layer(feats))

        grads = jax.grad(loss_fn)(features)
        assert grads.shape == features.shape
        # Gradients should be finite.
        assert jnp.all(jnp.isfinite(grads))

    def test_bandwidth_affects_output(self):
        features = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        layer_narrow = TopologicalLayer(num_eigvals=3, bandwidth=0.1)
        layer_wide = TopologicalLayer(num_eigvals=3, bandwidth=10.0)
        out_narrow = layer_narrow(features)
        out_wide = layer_wide(features)
        # Different bandwidths should give different outputs.
        assert not jnp.allclose(out_narrow, out_wide)


# ---------------------------------------------------------------------------
# hodge_laplacians
# ---------------------------------------------------------------------------


class TestHodgeLaplacians:
    """Test Hodge Laplacian computation."""

    def test_triangle_dimensions(self):
        """Triangle K3 has 3 vertices, 3 edges, 1 triangle."""
        hg = _triangle_hypergraph()
        laps = hodge_laplacians(hg, max_dim=2)
        assert len(laps) == 3

        # L_0: 3x3 (3 vertices)
        assert laps[0].shape == (3, 3)
        # L_1: 3x3 (3 edges)
        assert laps[1].shape == (3, 3)
        # L_2: 1x1 (1 triangle)
        assert laps[2].shape == (1, 1)

    def test_triangle_betti_numbers(self):
        """K3 filled triangle: beta_0=1, beta_1=1 (the triangle has a loop).

        But wait -- for the clique expansion of a single 3-hyperedge,
        we get K3 as a simplicial complex with the 2-simplex filled.
        With the 2-simplex filled: beta_0=1, beta_1=0, beta_2=0.
        Without it filled (just the boundary): beta_0=1, beta_1=1.

        Our clique expansion only gives the 1-skeleton (edges), so
        the 2-simplex is found via clique detection.  A single 3-hyperedge
        expands to K3 which is a complete graph on 3 vertices, so the
        2-simplex {0,1,2} IS detected.  Hence beta_1=0 for the filled
        complex.

        For the *boundary-only* triangle (3 edges, no face), we need
        to test with edges only.
        """
        # --- Filled triangle (single 3-hyperedge -> K3 with face) ---
        hg_filled = _triangle_hypergraph()
        laps = hodge_laplacians(hg_filled, max_dim=2)

        # beta_k = nullity(L_k)
        betti = []
        for L in laps:
            eigvals = np.linalg.eigvalsh(np.array(L))
            betti.append(int(np.sum(np.abs(eigvals) < 1e-6)))

        assert betti[0] == 1  # connected
        assert betti[1] == 0  # filled triangle kills the loop
        assert betti[2] == 0  # no cavities

    def test_square_cycle_betti(self):
        """Square cycle 0-1-2-3-0 (4 pairwise edges): beta_0=1, beta_1=1.

        The flag complex of a 4-cycle has no triangles (no 3-cliques),
        so the 1-cycle is not filled and beta_1=1.
        """
        hg = hgx.from_edge_list([(0, 1), (1, 2), (2, 3), (3, 0)])
        laps = hodge_laplacians(hg, max_dim=1)

        betti = []
        for L in laps:
            eigvals = np.linalg.eigvalsh(np.array(L))
            betti.append(int(np.sum(np.abs(eigvals) < 1e-6)))

        assert betti[0] == 1  # connected
        assert betti[1] == 1  # unfilled square has a loop

    def test_path_betti_numbers(self):
        """Path graph: beta_0=1, beta_1=0."""
        hg = _path_hypergraph()
        laps = hodge_laplacians(hg, max_dim=1)

        betti = []
        for L in laps:
            eigvals = np.linalg.eigvalsh(np.array(L))
            betti.append(int(np.sum(np.abs(eigvals) < 1e-6)))

        assert betti[0] == 1  # connected
        assert betti[1] == 0  # no loops

    def test_two_components_betti(self):
        """Two disconnected edges: beta_0=2."""
        hg = _two_components()
        laps = hodge_laplacians(hg, max_dim=1)

        eigvals = np.linalg.eigvalsh(np.array(laps[0]))
        beta_0 = int(np.sum(np.abs(eigvals) < 1e-6))
        assert beta_0 == 2

    def test_laplacians_symmetric(self):
        """Hodge Laplacians should be symmetric."""
        hg = _triangle_hypergraph()
        laps = hodge_laplacians(hg, max_dim=2)
        for L in laps:
            np.testing.assert_allclose(np.array(L), np.array(L).T, atol=1e-6)

    def test_laplacians_psd(self):
        """Hodge Laplacians should be positive semi-definite."""
        hg = _triangle_hypergraph()
        laps = hodge_laplacians(hg, max_dim=2)
        for L in laps:
            eigvals = np.linalg.eigvalsh(np.array(L))
            assert np.all(eigvals > -1e-6)

    def test_dimensions_match_boundary_ranks(self):
        """L_k should be square with size = number of k-simplices."""
        hg = _path_hypergraph()  # 4 nodes, 3 edges
        laps = hodge_laplacians(hg, max_dim=2)

        assert laps[0].shape[0] == 4  # 4 vertices
        assert laps[1].shape[0] == 3  # 3 edges
        # No triangles in a path graph.
        assert laps[2].shape[0] == 0 or laps[2].shape == (0, 0)


# ---------------------------------------------------------------------------
# compute_persistence (optional deps -- skip if not installed)
# ---------------------------------------------------------------------------


class TestComputePersistence:
    """Test compute_persistence (requires giotto-tda or ripser)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_backend(self):
        try:
            from gtda.homology import VietorisRipsPersistence  # noqa: F401
        except ImportError:
            try:
                import ripser  # noqa: F401
            except ImportError:
                pytest.skip("Neither giotto-tda nor ripser installed")

    def test_diagram_shapes(self):
        hg = _triangle_hypergraph()
        diagrams = hgx.compute_persistence(hg, filtration="clique", max_dim=1)
        assert len(diagrams) == 2  # dim 0 and dim 1
        for dgm in diagrams:
            assert dgm.ndim == 2
            assert dgm.shape[1] == 2

    def test_birth_le_death(self):
        hg = _path_hypergraph()
        diagrams = hgx.compute_persistence(hg, filtration="clique", max_dim=1)
        for dgm in diagrams:
            if len(dgm) > 0:
                assert np.all(dgm[:, 0] <= dgm[:, 1])

    def test_weight_filtration(self):
        hg = _triangle_hypergraph()
        diagrams = hgx.compute_persistence(hg, filtration="weight", max_dim=1)
        assert len(diagrams) == 2

    def test_degree_filtration(self):
        hg = _path_hypergraph()
        diagrams = hgx.compute_persistence(hg, filtration="degree", max_dim=1)
        assert len(diagrams) == 2


class TestPersistenceFeatures:
    """Test the persistence_features convenience wrapper."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_backend(self):
        try:
            from gtda.homology import VietorisRipsPersistence  # noqa: F401
        except ImportError:
            try:
                import ripser  # noqa: F401
            except ImportError:
                pytest.skip("Neither giotto-tda nor ripser installed")

    def test_landscape_features_shape(self):
        hg = _triangle_hypergraph()
        feats = hgx.persistence_features(
            hg, method="landscape", filtration="clique",
            max_dim=1, num_landscapes=5, resolution=100,
        )
        assert feats.shape == (2, 5, 100)

    def test_image_features_shape(self):
        hg = _triangle_hypergraph()
        feats = hgx.persistence_features(
            hg, method="image", filtration="clique",
            max_dim=1, resolution=50, sigma=0.1,
        )
        assert feats.shape == (2, 50, 50)
