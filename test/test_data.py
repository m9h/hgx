"""Tests for C. elegans data loaders.

Network-dependent tests are marked with ``pytest.mark.slow`` so they can
be skipped with ``pytest -m "not slow"``.  Offline tests exercise the
synthetic and hardcoded loaders without network access.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hgx._data import (
    _KARATE_EDGES,
    _KARATE_LABELS,
    _LINEAGE_DIVISIONS,
    load_cell_lineage,
    load_connectome,
    load_devograph,
    load_synthetic_karate,
)
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

slow = pytest.mark.slow


def _is_valid_hypergraph(hg: Hypergraph) -> None:
    """Assert basic structural invariants on a Hypergraph."""
    assert isinstance(hg, Hypergraph)
    n, m = hg.incidence.shape
    assert hg.node_features.shape[0] == n
    assert n > 0
    assert m > 0
    # Incidence entries should be 0 or 1.
    vals = np.unique(np.asarray(hg.incidence))
    assert set(vals.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 1. Connectome (network-dependent)
# ---------------------------------------------------------------------------


@slow
class TestLoadConnectomePairwise:
    """Test ``load_connectome(mode='pairwise')``."""

    @pytest.fixture(scope="class")
    def hg(self, tmp_path_factory):
        cache = tmp_path_factory.mktemp("hgx_cache")
        return load_connectome(mode="pairwise", cache_dir=cache)

    def test_is_hypergraph(self, hg):
        _is_valid_hypergraph(hg)

    def test_neuron_count(self, hg):
        # White 1986 whole connectome has ~280-300 unique neuron names.
        n = hg.incidence.shape[0]
        assert 200 <= n <= 400

    def test_pairwise_edges(self, hg):
        """Every hyperedge should contain 1 or 2 nodes (1 for self-synapses)."""
        edge_sizes = jnp.sum(hg.incidence, axis=0)
        assert jnp.all((edge_sizes == 1) | (edge_sizes == 2))

    def test_node_features_are_degrees(self, hg):
        n = hg.incidence.shape[0]
        assert hg.node_features.shape == (n, 1)
        # Degree should be positive for every node.
        assert jnp.all(hg.node_features > 0)

    def test_edge_features_present(self, hg):
        m = hg.incidence.shape[1]
        assert hg.edge_features is not None
        assert hg.edge_features.shape == (m, 1)
        # Synapse counts are positive integers.
        assert jnp.all(hg.edge_features >= 1)


@slow
class TestLoadConnectomeGrouped:
    """Test ``load_connectome(mode='grouped')``."""

    @pytest.fixture(scope="class")
    def hg(self, tmp_path_factory):
        cache = tmp_path_factory.mktemp("hgx_cache")
        return load_connectome(mode="grouped", cache_dir=cache)

    def test_is_hypergraph(self, hg):
        _is_valid_hypergraph(hg)

    def test_three_groups(self, hg):
        """There should be exactly 3 hyperedges (sensory, inter, motor)."""
        assert hg.incidence.shape[1] == 3

    def test_one_hot_features(self, hg):
        n = hg.incidence.shape[0]
        assert hg.node_features.shape == (n, n)

    def test_every_neuron_in_a_group(self, hg):
        """Each neuron should belong to at least one group."""
        membership = jnp.sum(hg.incidence, axis=1)
        assert jnp.all(membership >= 1)


def test_connectome_bad_mode(tmp_path):
    """Mode validation happens before download, so no network needed."""
    with pytest.raises(ValueError, match="Unknown mode"):
        load_connectome(mode="nonexistent", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# 2. Cell lineage (offline -- hardcoded data)
# ---------------------------------------------------------------------------


class TestLoadCellLineage:
    """Test ``load_cell_lineage()`` -- fully offline."""

    def test_is_hypergraph(self):
        hg = load_cell_lineage()
        _is_valid_hypergraph(hg)

    def test_3_uniform(self):
        """Every hyperedge should contain exactly 3 cells."""
        hg = load_cell_lineage()
        edge_sizes = jnp.sum(hg.incidence, axis=0)
        assert jnp.all(edge_sizes == 3)

    def test_full_lineage_counts(self):
        """Full lineage should have 30 divisions, 61 unique cells."""
        hg = load_cell_lineage()
        assert hg.incidence.shape[1] == len(_LINEAGE_DIVISIONS)
        assert hg.incidence.shape[1] == 30
        assert hg.incidence.shape[0] == 61

    def test_one_hot_features(self):
        hg = load_cell_lineage()
        n = hg.incidence.shape[0]
        assert hg.node_features.shape == (n, n)
        assert jnp.allclose(hg.node_features, jnp.eye(n))

    def test_max_depth_1(self):
        """Depth 1: only the zygote division P0 -> AB + P1."""
        hg = load_cell_lineage(max_depth=1)
        assert hg.incidence.shape[1] == 1  # 1 division
        assert hg.incidence.shape[0] == 3  # P0, AB, P1

    def test_max_depth_2(self):
        """Depth 2: rounds 1 + 2 = 1 + 2 = 3 divisions."""
        hg = load_cell_lineage(max_depth=2)
        assert hg.incidence.shape[1] == 3

    def test_max_depth_3(self):
        """Depth 3: rounds 1-3 = 1 + 2 + 4 = 7 divisions."""
        hg = load_cell_lineage(max_depth=3)
        assert hg.incidence.shape[1] == 7


# ---------------------------------------------------------------------------
# 3. DevoGraph (network-dependent)
# ---------------------------------------------------------------------------


@slow
class TestLoadDevographAll:
    """Test ``load_devograph()`` returning all time steps."""

    @pytest.fixture(scope="class")
    def hgs(self, tmp_path_factory):
        cache = tmp_path_factory.mktemp("hgx_cache")
        return load_devograph(cache_dir=cache)

    def test_returns_list(self, hgs):
        assert isinstance(hgs, list)
        assert len(hgs) > 0

    def test_time_step_count(self, hgs):
        """Data has ~190+ time steps."""
        assert len(hgs) >= 190

    def test_first_frame_single_cell(self, hgs):
        """Time step 1 has only the AB cell."""
        hg = hgs[0]
        _is_valid_hypergraph(hg)
        assert hg.incidence.shape[0] == 1

    def test_feature_dim(self, hgs):
        """Node features should be 4-d: [x, y, z, size]."""
        for hg in hgs[:5]:
            assert hg.node_dim == 4

    def test_positions_present(self, hgs):
        for hg in hgs[:5]:
            assert hg.positions is not None
            assert hg.positions.shape == (hg.incidence.shape[0], 3)

    def test_geometry_euclidean(self, hgs):
        for hg in hgs[:5]:
            assert hg.geometry == "euclidean"


@slow
class TestLoadDevographSingle:
    """Test ``load_devograph(time_step=...)``."""

    @pytest.fixture(scope="class")
    def cache(self, tmp_path_factory):
        return tmp_path_factory.mktemp("hgx_cache")

    def test_single_step(self, cache):
        hg = load_devograph(time_step=10, cache_dir=cache)
        assert isinstance(hg, Hypergraph)
        _is_valid_hypergraph(hg)

    def test_bad_step(self, cache):
        with pytest.raises(ValueError, match="not found"):
            load_devograph(time_step=9999, cache_dir=cache)


# ---------------------------------------------------------------------------
# 4. Synthetic karate (offline)
# ---------------------------------------------------------------------------


class TestLoadSyntheticKarate:
    """Test ``load_synthetic_karate()`` -- fully offline."""

    def test_is_hypergraph(self):
        hg = load_synthetic_karate()
        _is_valid_hypergraph(hg)

    def test_34_nodes(self):
        hg = load_synthetic_karate()
        assert hg.incidence.shape[0] == 34

    def test_node_features_one_hot(self):
        hg = load_synthetic_karate()
        assert hg.node_features.shape == (34, 2)
        # Each row should sum to 1 (one-hot).
        row_sums = jnp.sum(hg.node_features, axis=1)
        assert jnp.allclose(row_sums, 1.0)

    def test_community_labels(self):
        """Node features should encode the known community split."""
        hg = load_synthetic_karate()
        labels = np.argmax(np.asarray(hg.node_features), axis=1)
        assert labels.tolist() == _KARATE_LABELS

    def test_has_cliques_larger_than_2(self):
        """The clique lifting should produce some hyperedges of size > 2."""
        hg = load_synthetic_karate()
        edge_sizes = jnp.sum(hg.incidence, axis=0)
        assert jnp.max(edge_sizes) > 2

    def test_all_edges_covered(self):
        """Every original edge should appear within some hyperedge."""
        hg = load_synthetic_karate()
        H = np.asarray(hg.incidence)
        for i, j in _KARATE_EDGES:
            # There should be at least one hyperedge containing both i and j.
            both = H[i, :] * H[j, :]
            assert np.any(both > 0), f"Edge ({i}, {j}) not covered"

    def test_every_node_in_some_edge(self):
        hg = load_synthetic_karate()
        membership = jnp.sum(hg.incidence, axis=1)
        assert jnp.all(membership >= 1)
