"""Tests for gene regulatory network data loaders (general-purpose).

All tests are offline, using synthetic data generated in fixtures.
No network access or external files are required.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hgx._grn import (
    grn_to_temporal_hypergraphs,
    load_grn_from_csv,
    load_grn_from_edge_list,
)
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_hypergraph(hg: Hypergraph) -> None:
    """Assert basic structural invariants."""
    assert isinstance(hg, Hypergraph)
    n, m = hg.incidence.shape
    assert hg.node_features.shape[0] == n
    assert n > 0
    assert m > 0
    vals = np.unique(np.asarray(hg.incidence))
    assert set(vals.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Synthetic edge data
# ---------------------------------------------------------------------------

# A small synthetic GRN:
#   TF1 -> geneA (0.5), TF1 -> geneB (0.3)
#   TF2 -> geneB (0.8), TF2 -> geneC (-0.2)
#   TF3 -> geneA (0.1)
_EDGES: list[tuple[str, str, float]] = [
    ("TF1", "geneA", 0.5),
    ("TF1", "geneB", 0.3),
    ("TF2", "geneB", 0.8),
    ("TF2", "geneC", -0.2),
    ("TF3", "geneA", 0.1),
]

_MODULES = {
    "mod1": ["TF1", "geneA", "geneB"],
    "mod2": ["TF2", "geneB", "geneC"],
}


# ---------------------------------------------------------------------------
# 1. load_grn_from_edge_list
# ---------------------------------------------------------------------------


class TestLoadGRNFromEdgeList:
    """Test ``load_grn_from_edge_list``."""

    def test_grouped_by_tf(self):
        """Without modules, one hyperedge per TF regulon."""
        hg = load_grn_from_edge_list(_EDGES)
        _is_valid_hypergraph(hg)
        # 3 TFs -> 3 hyperedges
        assert hg.incidence.shape[1] == 3

    def test_node_count(self):
        """Should have 5 unique genes: TF1, TF2, TF3, geneA, geneB, geneC."""
        hg = load_grn_from_edge_list(_EDGES)
        assert hg.incidence.shape[0] == 6

    def test_tf_in_own_regulon(self):
        """Each TF should be a member of its own regulon hyperedge."""
        hg = load_grn_from_edge_list(_EDGES)
        # Every hyperedge should contain at least 2 nodes (TF + targets)
        edge_sizes = jnp.sum(hg.incidence, axis=0)
        assert jnp.all(edge_sizes >= 2)

    def test_degree_features(self):
        """Node features should be degree vectors of shape (n, 1)."""
        hg = load_grn_from_edge_list(_EDGES)
        n = hg.incidence.shape[0]
        assert hg.node_features.shape == (n, 1)
        # Every node should be in at least one hyperedge
        assert jnp.all(hg.node_features >= 1)

    def test_edge_features(self):
        """Edge features should be aggregated weights (m, 1)."""
        hg = load_grn_from_edge_list(_EDGES)
        m = hg.incidence.shape[1]
        assert hg.edge_features is not None
        assert hg.edge_features.shape == (m, 1)
        # All weights should be non-negative (absolute values)
        assert jnp.all(hg.edge_features >= 0)

    def test_with_modules(self):
        """With explicit modules, one hyperedge per module."""
        hg = load_grn_from_edge_list(_EDGES, modules=_MODULES)
        _is_valid_hypergraph(hg)
        assert hg.incidence.shape[1] == 2  # mod1, mod2

    def test_module_membership(self):
        """Module members should be in the correct hyperedge."""
        hg = load_grn_from_edge_list(_EDGES, modules=_MODULES)
        # mod1 has 3 members, mod2 has 3 members
        edge_sizes = jnp.sum(hg.incidence, axis=0)
        assert jnp.all(edge_sizes == 3)

    def test_single_edge(self):
        """Works with a single regulatory edge."""
        edges = [("TF1", "geneA", 1.0)]
        hg = load_grn_from_edge_list(edges)
        _is_valid_hypergraph(hg)
        assert hg.incidence.shape == (2, 1)

    def test_overlapping_modules(self):
        """Genes can belong to multiple modules."""
        mods = {
            "m1": ["A", "B", "C"],
            "m2": ["B", "C", "D"],
        }
        edges = [("A", "B", 1.0), ("C", "D", 1.0)]
        hg = load_grn_from_edge_list(edges, modules=mods)
        # Gene B and C are in both modules
        degrees = jnp.sum(hg.incidence, axis=1)
        assert jnp.max(degrees) >= 2


# ---------------------------------------------------------------------------
# 2. load_grn_from_csv
# ---------------------------------------------------------------------------


class TestLoadGRNFromCSV:
    """Test ``load_grn_from_csv``."""

    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        """Write a synthetic GRN CSV."""
        p = tmp_path / "grn.csv"
        with open(p, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tf", "target", "weight"])
            for tf, target, w in _EDGES:
                writer.writerow([tf, target, w])
        return p

    @pytest.fixture
    def csv_with_modules(self, tmp_path: Path) -> Path:
        """Write a synthetic GRN CSV with module column."""
        p = tmp_path / "grn_modules.csv"
        with open(p, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tf", "target", "weight", "module"])
            writer.writerow(["TF1", "geneA", 0.5, "mod1"])
            writer.writerow(["TF1", "geneB", 0.3, "mod1"])
            writer.writerow(["TF2", "geneB", 0.8, "mod2"])
            writer.writerow(["TF2", "geneC", -0.2, "mod2"])
        return p

    def test_basic_load(self, csv_path):
        hg = load_grn_from_csv(csv_path)
        _is_valid_hypergraph(hg)
        assert hg.incidence.shape[0] == 6  # 6 genes
        assert hg.incidence.shape[1] == 3  # 3 TFs

    def test_custom_columns(self, tmp_path):
        p = tmp_path / "custom.csv"
        with open(p, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["regulator", "gene", "coeff"])
            writer.writerow(["TF1", "A", 0.5])
            writer.writerow(["TF1", "B", 0.3])
        hg = load_grn_from_csv(
            p, tf_col="regulator", target_col="gene", weight_col="coeff",
        )
        _is_valid_hypergraph(hg)
        assert hg.incidence.shape[0] == 3

    def test_missing_weight_col(self, tmp_path):
        """Missing weight column should default to 1.0."""
        p = tmp_path / "no_weight.csv"
        with open(p, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tf", "target"])
            writer.writerow(["TF1", "A"])
            writer.writerow(["TF1", "B"])
        hg = load_grn_from_csv(p)
        _is_valid_hypergraph(hg)
        assert hg.edge_features is not None
        # Default weight is 1.0
        assert jnp.allclose(hg.edge_features, jnp.array([[1.0]]))

    def test_with_module_col(self, csv_with_modules):
        hg = load_grn_from_csv(csv_with_modules, module_col="module")
        _is_valid_hypergraph(hg)
        assert hg.incidence.shape[1] == 2  # mod1, mod2

    def test_missing_column_error(self, tmp_path):
        p = tmp_path / "bad.csv"
        with open(p, "w", newline="") as fh:
            fh.write("col1,col2\na,b\n")
        with pytest.raises(ValueError, match="not found"):
            load_grn_from_csv(p)

    def test_empty_csv_error(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_grn_from_csv(p)

    def test_tab_delimited(self, tmp_path):
        """TSV files should work too."""
        p = tmp_path / "grn.tsv"
        with open(p, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(["tf", "target", "weight"])
            writer.writerow(["TF1", "A", 0.5])
            writer.writerow(["TF2", "B", 0.8])
        hg = load_grn_from_csv(p)
        _is_valid_hypergraph(hg)


# ---------------------------------------------------------------------------
# 3. load_grn_from_anndata (skip if anndata not installed)
# ---------------------------------------------------------------------------

try:
    import anndata  # noqa: F401

    _HAS_ANNDATA = True
except ImportError:
    _HAS_ANNDATA = False

requires_anndata = pytest.mark.skipif(
    not _HAS_ANNDATA, reason="anndata not installed",
)


@requires_anndata
class TestLoadGRNFromAnndata:
    """Test ``load_grn_from_anndata``."""

    @pytest.fixture
    def h5ad_path(self, tmp_path: Path) -> Path:
        """Create a tiny synthetic AnnData file."""
        import anndata

        rng = np.random.default_rng(42)
        n_cells, n_genes = 50, 10
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        obs = {"cell_type": ["typeA"] * 25 + ["typeB"] * 25}
        var = {"module": [f"mod{i % 3}" for i in range(n_genes)]}
        adata = anndata.AnnData(
            X=X,
            obs=obs,
            var=var,
        )
        adata.var_names = [f"gene{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell{i}" for i in range(n_cells)]
        p = tmp_path / "test.h5ad"
        adata.write_h5ad(p)
        return p

    def test_basic_load(self, h5ad_path):
        from hgx._grn import load_grn_from_anndata

        hg = load_grn_from_anndata(h5ad_path)
        _is_valid_hypergraph(hg)
        assert hg.incidence.shape[0] == 10  # 10 genes

    def test_with_module_key(self, h5ad_path):
        from hgx._grn import load_grn_from_anndata

        hg = load_grn_from_anndata(h5ad_path, module_key="module")
        _is_valid_hypergraph(hg)
        # 3 modules: mod0, mod1, mod2
        assert hg.incidence.shape[1] == 3

    def test_with_obs_key(self, h5ad_path):
        from hgx._grn import load_grn_from_anndata

        hg = load_grn_from_anndata(h5ad_path, obs_key="cell_type")
        # Node features should be (n_genes, 2) — one col per cell type
        assert hg.node_features.shape == (10, 2)

    def test_default_features(self, h5ad_path):
        from hgx._grn import load_grn_from_anndata

        hg = load_grn_from_anndata(h5ad_path)
        # Without obs_key, features are (n_genes, 1) — mean expr
        assert hg.node_features.shape == (10, 1)


# ---------------------------------------------------------------------------
# 4. grn_to_temporal_hypergraphs
# ---------------------------------------------------------------------------


class TestGRNToTemporalHypergraphs:
    """Test ``grn_to_temporal_hypergraphs``."""

    def test_basic(self):
        rng = np.random.default_rng(0)
        n_cells, n_genes, n_edges = 60, 5, 3
        expr = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        times = np.array([0] * 20 + [1] * 20 + [2] * 20)
        H = rng.integers(0, 2, (n_genes, n_edges)).astype(np.float32)

        hgs = grn_to_temporal_hypergraphs(expr, times, H)
        assert len(hgs) == 3
        for hg in hgs:
            assert isinstance(hg, Hypergraph)
            assert hg.incidence.shape == (n_genes, n_edges)
            assert hg.node_features.shape == (n_genes, 1)

    def test_shared_incidence(self):
        """All timepoints should share the same incidence structure."""
        rng = np.random.default_rng(1)
        n_cells, n_genes = 40, 4
        expr = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        times = np.array([0] * 20 + [1] * 20)
        H = np.array([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)

        hgs = grn_to_temporal_hypergraphs(expr, times, H)
        assert jnp.array_equal(hgs[0].incidence, hgs[1].incidence)

    def test_different_features(self):
        """Different timepoints should have different node features."""
        n_cells, n_genes = 40, 4
        # Make expression clearly different between timepoints
        expr = np.zeros((n_cells, n_genes), dtype=np.float32)
        expr[:20] = 1.0  # time 0: all ones
        expr[20:] = 5.0  # time 1: all fives
        times = np.array([0] * 20 + [1] * 20)
        H = np.eye(n_genes, dtype=np.float32)

        hgs = grn_to_temporal_hypergraphs(expr, times, H)
        assert not jnp.allclose(hgs[0].node_features, hgs[1].node_features)
        assert jnp.allclose(hgs[0].node_features, 1.0)
        assert jnp.allclose(hgs[1].node_features, 5.0)

    def test_num_timepoints(self):
        """num_timepoints limits the output."""
        rng = np.random.default_rng(3)
        expr = rng.standard_normal((60, 5)).astype(np.float32)
        times = np.array([0] * 20 + [1] * 20 + [2] * 20)
        H = np.eye(5, dtype=np.float32)

        hgs = grn_to_temporal_hypergraphs(expr, times, H, num_timepoints=2)
        assert len(hgs) == 2

    def test_float_times(self):
        """Float time labels should work."""
        rng = np.random.default_rng(4)
        expr = rng.standard_normal((40, 3)).astype(np.float32)
        times = np.array([0.0] * 10 + [0.5] * 10 + [1.0] * 10 + [1.5] * 10)
        H = np.eye(3, dtype=np.float32)

        hgs = grn_to_temporal_hypergraphs(expr, times, H)
        assert len(hgs) == 4

    def test_sorted_by_time(self):
        """Output should be sorted by time even if input is shuffled."""
        expr = np.zeros((30, 3), dtype=np.float32)
        # Shuffle: time 2 first, then 0, then 1
        expr[:10] = 3.0  # time 2
        expr[10:20] = 1.0  # time 0
        expr[20:] = 2.0  # time 1
        times = np.array([2] * 10 + [0] * 10 + [1] * 10)
        H = np.eye(3, dtype=np.float32)

        hgs = grn_to_temporal_hypergraphs(expr, times, H)
        assert len(hgs) == 3
        # Sorted: time 0=1.0, time 1=2.0, time 2=3.0
        assert jnp.allclose(hgs[0].node_features, 1.0)
        assert jnp.allclose(hgs[1].node_features, 2.0)
        assert jnp.allclose(hgs[2].node_features, 3.0)

    def test_mismatched_shapes(self):
        """Incompatible shapes should raise ValueError."""
        expr = np.zeros((10, 5), dtype=np.float32)
        times = np.zeros(8)  # wrong length
        H = np.eye(5, dtype=np.float32)
        with pytest.raises(ValueError, match="time_labels length"):
            grn_to_temporal_hypergraphs(expr, times, H)

    def test_mismatched_incidence(self):
        """Wrong incidence rows should raise ValueError."""
        expr = np.zeros((10, 5), dtype=np.float32)
        times = np.zeros(10)
        H = np.eye(3, dtype=np.float32)  # 3 != 5
        with pytest.raises(ValueError, match="incidence rows"):
            grn_to_temporal_hypergraphs(expr, times, H)
