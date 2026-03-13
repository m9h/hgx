"""Gene regulatory network (GRN) data loaders for hypergraph neural networks.

Converts gene regulatory networks into :class:`~hgx.Hypergraph` objects.
Supports edge-list input, CSV files (e.g. Pando output), AnnData .h5ad
files, and temporal expression data for Neural ODE training.

Core loaders use only stdlib + JAX + NumPy.  The AnnData loader requires
the optional ``anndata`` extra: ``pip install hgx[anndata]``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from hgx._hypergraph import from_incidence, Hypergraph


# ---------------------------------------------------------------------------
# 1. From edge list
# ---------------------------------------------------------------------------


def load_grn_from_edge_list(
    edges: list[tuple[str, str, float]],
    modules: dict[str, list[str]] | None = None,
) -> Hypergraph:
    """Convert a GRN edge list into a Hypergraph.

    Each edge is a ``(tf, target, weight)`` tuple representing a
    regulatory interaction.

    Args:
        edges: List of ``(transcription_factor, target_gene, weight)``
            tuples.
        modules: Optional dict mapping ``module_id -> [gene_name, ...]``.
            If provided, each module becomes one hyperedge.  Otherwise,
            edges are grouped by source TF (all targets of one TF form
            one hyperedge, with the TF itself also included).

    Returns:
        A :class:`~hgx.Hypergraph`.  Node features are degree vectors
        (shape ``(n, 1)``).  Edge features are aggregated absolute
        weights per hyperedge (shape ``(m, 1)``).
    """
    # Collect unique gene names
    genes: dict[str, int] = {}
    for tf, target, _w in edges:
        genes.setdefault(tf, len(genes))
        genes.setdefault(target, len(genes))
    n = len(genes)

    if modules is not None:
        # Modules define hyperedges
        # Add any genes in modules that weren't in edges
        for members in modules.values():
            for g in members:
                genes.setdefault(g, len(genes))
        n = len(genes)

        module_ids = sorted(modules.keys())
        m = len(module_ids)
        H = np.zeros((n, m), dtype=np.float32)
        edge_weights = np.zeros((m, 1), dtype=np.float32)

        # Index edges by gene pair for weight lookup
        weight_map: dict[tuple[str, str], float] = {}
        for tf, target, w in edges:
            weight_map[(tf, target)] = w

        for k, mid in enumerate(module_ids):
            members = modules[mid]
            for g in members:
                if g in genes:
                    H[genes[g], k] = 1.0
            # Edge weight: mean absolute weight of edges within module
            abs_weights = []
            for g in members:
                for tf, target, w in edges:
                    if tf in members and target == g:
                        abs_weights.append(abs(w))
                    elif target in members and tf == g:
                        abs_weights.append(abs(w))
            edge_weights[k, 0] = float(np.mean(abs_weights)) if abs_weights else 0.0
    else:
        # Group by source TF: each TF's regulon is a hyperedge
        tf_targets: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for tf, target, w in edges:
            tf_targets[tf].append((target, w))

        tf_list = sorted(tf_targets.keys())
        m = len(tf_list)
        H = np.zeros((n, m), dtype=np.float32)
        edge_weights = np.zeros((m, 1), dtype=np.float32)

        for k, tf in enumerate(tf_list):
            H[genes[tf], k] = 1.0  # TF is in its own regulon
            targets_weights = tf_targets[tf]
            for target, _w in targets_weights:
                H[genes[target], k] = 1.0
            edge_weights[k, 0] = float(
                np.mean([abs(w) for _, w in targets_weights])
            )

    # Node features: degree
    degrees = H.sum(axis=1, keepdims=True)

    return from_incidence(
        jnp.array(H),
        node_features=jnp.array(degrees),
        edge_features=jnp.array(edge_weights),
    )


# ---------------------------------------------------------------------------
# 2. From CSV
# ---------------------------------------------------------------------------


def load_grn_from_csv(
    path: str | Path,
    *,
    tf_col: str = "tf",
    target_col: str = "target",
    weight_col: str = "weight",
    module_col: str | None = None,
) -> Hypergraph:
    """Load a GRN from a CSV file of regulatory coefficients.

    Compatible with Pando's ``coef()`` output and similar tabular
    GRN formats.

    Args:
        path: Path to the CSV file.
        tf_col: Column name for transcription factor names.
        target_col: Column name for target gene names.
        weight_col: Column name for interaction weight/coefficient.
            If the column is missing, all weights default to 1.0.
        module_col: Optional column name for module/group IDs.  If
            provided, edges are grouped into modules as hyperedges.

    Returns:
        A :class:`~hgx.Hypergraph` (see :func:`load_grn_from_edge_list`).
    """
    path = Path(path)
    with open(path, newline="") as fh:
        sample = fh.read(4096)
        if not sample.strip():
            raise ValueError(f"CSV file is empty: {path}")
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(fh, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")

    # Validate required columns
    available = set(rows[0].keys())
    for col_name, col_label in [(tf_col, "tf"), (target_col, "target")]:
        if col_name not in available:
            raise ValueError(
                f"Column {col_label}={col_name!r} not found. "
                f"Available columns: {sorted(available)}"
            )

    has_weight = weight_col in available

    edges: list[tuple[str, str, float]] = []
    modules: dict[str, list[str]] | None = None

    if module_col is not None and module_col in available:
        modules = defaultdict(list)

    gene_set_per_module: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        tf = row[tf_col].strip()
        target = row[target_col].strip()
        w = float(row[weight_col]) if has_weight else 1.0
        edges.append((tf, target, w))

        if modules is not None:
            mid = row[module_col].strip()
            gene_set_per_module[mid].add(tf)
            gene_set_per_module[mid].add(target)

    if modules is not None:
        modules = {mid: sorted(genes) for mid, genes in gene_set_per_module.items()}

    return load_grn_from_edge_list(edges, modules=modules)


# ---------------------------------------------------------------------------
# 3. From AnnData
# ---------------------------------------------------------------------------


def load_grn_from_anndata(
    path: str | Path,
    *,
    layer: str = "X",
    obs_key: str | None = None,
    var_names_key: str | None = None,
    module_key: str | None = None,
    n_modules: int = 10,
    corr_threshold: float = 0.3,
) -> Hypergraph:
    """Load a GRN from an AnnData ``.h5ad`` file.

    Nodes are genes (``var``).  Node features are mean expression per
    gene across cells (or per cell-type group if *obs_key* is given,
    yielding a multi-dimensional feature vector).

    Hyperedges are built from gene modules.  If *module_key* names a
    column in ``adata.var``, those pre-computed module assignments are
    used.  Otherwise, modules are derived by Leiden clustering on the
    gene-gene correlation graph (thresholded at *corr_threshold*).

    Requires the ``anndata`` extra: ``pip install hgx[anndata]``.

    Args:
        path: Path to the ``.h5ad`` file.
        layer: Which data layer to use (``"X"`` for default).
        obs_key: Optional column in ``adata.obs`` to group cells
            (e.g. ``"cell_type"``).  Produces one feature dimension
            per group (mean expression).
        var_names_key: Optional column in ``adata.var`` for gene names.
            Defaults to ``adata.var_names``.
        module_key: Optional column in ``adata.var`` containing
            pre-computed module IDs for each gene.
        n_modules: Number of Leiden clusters when auto-detecting modules.
        corr_threshold: Minimum absolute correlation for the gene-gene
            graph used in Leiden clustering.

    Returns:
        A :class:`~hgx.Hypergraph`.
    """
    try:
        import anndata
    except ImportError as e:
        raise ImportError(
            "load_grn_from_anndata requires anndata. "
            "Install with: pip install hgx[anndata]"
        ) from e

    adata = anndata.read_h5ad(Path(path))

    # Extract expression matrix
    if layer == "X":
        X = np.asarray(adata.X)
    else:
        X = np.asarray(adata.layers[layer])

    # Ensure dense
    try:
        from scipy.sparse import issparse

        if issparse(X):
            X = np.asarray(X.toarray())
    except ImportError:
        pass

    X = X.astype(np.float32)
    n_genes = X.shape[1]

    # Build node features: mean expression
    if obs_key is not None and obs_key in adata.obs.columns:
        groups = adata.obs[obs_key]
        unique_groups = sorted(groups.unique())
        feats = np.zeros((n_genes, len(unique_groups)), dtype=np.float32)
        for i, g in enumerate(unique_groups):
            mask = (groups == g).values
            feats[:, i] = X[mask].mean(axis=0)
    else:
        feats = X.mean(axis=0, keepdims=True).T  # (n_genes, 1)

    # Build modules
    if module_key is not None and module_key in adata.var.columns:
        # Pre-computed modules
        module_labels = adata.var[module_key].values
        modules: dict[str, list[int]] = defaultdict(list)
        for i, label in enumerate(module_labels):
            modules[str(label)].append(i)
    else:
        # Auto-detect via correlation clustering
        modules = _correlation_modules(X, n_modules, corr_threshold)

    # Build incidence
    module_list = sorted(modules.keys())
    m = len(module_list)
    H = np.zeros((n_genes, m), dtype=np.float32)
    for k, mid in enumerate(module_list):
        for idx in modules[mid]:
            H[idx, k] = 1.0

    return from_incidence(
        jnp.array(H),
        node_features=jnp.array(feats),
    )


def _correlation_modules(
    X: np.ndarray,
    n_modules: int,
    threshold: float,
) -> dict[str, list[int]]:
    """Cluster genes into modules via correlation thresholding + Leiden.

    Falls back to simple greedy connected-component clustering if
    ``scanpy`` is not available.
    """
    n_genes = X.shape[1]

    # Compute gene-gene correlation
    X_centered = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X_centered, axis=0, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    X_normed = X_centered / norms
    corr = (X_normed.T @ X_normed) / max(X.shape[0], 1)  # (genes, genes)

    # Build adjacency from correlation threshold
    adj = (np.abs(corr) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)

    try:
        # Use scanpy's Leiden clustering on the correlation graph
        import anndata
        import scanpy as sc

        gene_adata = anndata.AnnData(
            X=np.eye(n_genes, dtype=np.float32),
            obsp={"connectivities": __import__("scipy").sparse.csr_matrix(adj)},
        )
        # Set neighbors flag so Leiden doesn't recompute
        gene_adata.uns["neighbors"] = {"connectivities_key": "connectivities"}
        resolution = max(0.1, n_modules / 10.0)
        sc.tl.leiden(gene_adata, resolution=resolution, key_added="module")
        labels = gene_adata.obs["module"].values
        modules: dict[str, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            modules[str(label)].append(i)
        return dict(modules)
    except ImportError:
        pass

    # Fallback: simple greedy connected components with size limit
    return _greedy_modules(adj, n_genes, n_modules)


def _greedy_modules(
    adj: np.ndarray,
    n_genes: int,
    n_modules: int,
) -> dict[str, list[int]]:
    """Simple connected-component-based module detection fallback."""
    visited = np.zeros(n_genes, dtype=bool)
    modules: dict[str, list[int]] = {}
    module_id = 0

    for start in range(n_genes):
        if visited[start]:
            continue
        # BFS
        queue = [start]
        component: list[int] = []
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            neighbors = np.where(adj[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    queue.append(nb)

        modules[str(module_id)] = component
        module_id += 1

    # If too many modules, merge smallest ones
    if len(modules) > n_modules:
        sorted_mods = sorted(modules.items(), key=lambda x: len(x[1]), reverse=True)
        merged: dict[str, list[int]] = {}
        for i, (_, members) in enumerate(sorted_mods):
            bucket = str(i % n_modules)
            merged.setdefault(bucket, []).extend(members)
        return merged

    return modules


# ---------------------------------------------------------------------------
# 4. Pando-specific loader
# ---------------------------------------------------------------------------


def load_pando_modules(
    coef_csv: str | Path,
    modules_csv: str | Path | None = None,
    *,
    padj_threshold: float = 0.05,
    tf_col: str = "tf",
    target_col: str = "target",
    estimate_col: str = "estimate",
    padj_col: str = "padj",
    module_col: str = "module",
    gene_col: str = "gene",
) -> Hypergraph:
    """Load Pando GRN inference output as a Hypergraph.

    Reads a coefficients CSV (as produced by Pando's ``coef()``
    method), filters to significant edges (``padj < padj_threshold``),
    and builds a hypergraph.

    Args:
        coef_csv: Path to the Pando coefficients CSV with columns
            for TF, target, estimate (weight), and adjusted p-value.
        modules_csv: Optional path to a CSV mapping genes to module
            IDs (columns: *gene_col*, *module_col*).  If provided,
            modules define hyperedges.  Otherwise, edges are grouped
            by source TF.
        padj_threshold: Maximum adjusted p-value for inclusion.
        tf_col: Column name for TF in *coef_csv*.
        target_col: Column name for target gene in *coef_csv*.
        estimate_col: Column name for effect estimate (weight).
        padj_col: Column name for adjusted p-value.
        module_col: Column name for module ID in *modules_csv*.
        gene_col: Column name for gene name in *modules_csv*.

    Returns:
        A :class:`~hgx.Hypergraph`.
    """
    coef_csv = Path(coef_csv)
    with open(coef_csv, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(fh, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Pando coefficients CSV is empty: {coef_csv}")

    available = set(rows[0].keys())
    for col_name, label in [
        (tf_col, "tf"),
        (target_col, "target"),
        (estimate_col, "estimate"),
        (padj_col, "padj"),
    ]:
        if col_name not in available:
            raise ValueError(
                f"Column {label}={col_name!r} not found. "
                f"Available: {sorted(available)}"
            )

    # Filter to significant edges
    edges: list[tuple[str, str, float]] = []
    for row in rows:
        padj = float(row[padj_col])
        if padj < padj_threshold:
            tf = row[tf_col].strip()
            target = row[target_col].strip()
            w = float(row[estimate_col])
            edges.append((tf, target, w))

    if not edges:
        raise ValueError(
            f"No significant edges found (padj < {padj_threshold}) "
            f"in {coef_csv}"
        )

    # Load optional modules mapping
    modules: dict[str, list[str]] | None = None
    if modules_csv is not None:
        modules_csv = Path(modules_csv)
        with open(modules_csv, newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            reader = csv.DictReader(fh, dialect=dialect)
            mod_rows = list(reader)

        mod_available = set(mod_rows[0].keys()) if mod_rows else set()
        if gene_col not in mod_available or module_col not in mod_available:
            raise ValueError(
                f"Modules CSV must have columns {gene_col!r} and "
                f"{module_col!r}. Found: {sorted(mod_available)}"
            )

        mod_groups: dict[str, list[str]] = defaultdict(list)
        for row in mod_rows:
            mid = row[module_col].strip()
            gene = row[gene_col].strip()
            mod_groups[mid].append(gene)
        modules = dict(mod_groups)

    return load_grn_from_edge_list(edges, modules=modules)


# ---------------------------------------------------------------------------
# 5. Temporal hypergraphs from expression data
# ---------------------------------------------------------------------------


def grn_to_temporal_hypergraphs(
    expression_matrix: Any,
    time_labels: Any,
    incidence: Any,
    *,
    num_timepoints: int | None = None,
) -> list[Hypergraph]:
    """Build temporal hypergraphs from expression data and a fixed GRN.

    Given a ``(cells, genes)`` expression matrix, per-cell time labels,
    and a fixed incidence structure (e.g. from a GRN loader), produces
    one :class:`~hgx.Hypergraph` per timepoint where node features are
    mean expression of each gene at that timepoint.

    Useful for training Neural ODE/CDE models on developmental
    trajectories where the regulatory structure is static but gene
    expression evolves.

    Args:
        expression_matrix: Array-like of shape ``(n_cells, n_genes)``.
            Dense numpy or JAX array.
        time_labels: Array-like of shape ``(n_cells,)`` with a time
            label for each cell (integer or float).
        incidence: Array-like of shape ``(n_genes, n_edges)`` — the
            shared hypergraph incidence matrix.
        num_timepoints: If given, only include the first *num_timepoints*
            unique sorted timepoints.

    Returns:
        List of :class:`~hgx.Hypergraph` objects, one per timepoint,
        sorted by time.  Each has node features of shape
        ``(n_genes, 1)`` (mean expression at that time).
    """
    expr = np.asarray(expression_matrix, dtype=np.float32)
    times = np.asarray(time_labels)
    H = np.asarray(incidence, dtype=np.float32)

    n_cells, n_genes = expr.shape
    if times.shape[0] != n_cells:
        raise ValueError(
            f"time_labels length ({times.shape[0]}) must match "
            f"expression_matrix rows ({n_cells})"
        )
    if H.shape[0] != n_genes:
        raise ValueError(
            f"incidence rows ({H.shape[0]}) must match "
            f"expression_matrix columns ({n_genes})"
        )

    unique_times = sorted(set(times.tolist()))
    if num_timepoints is not None:
        unique_times = unique_times[:num_timepoints]

    H_jnp = jnp.array(H)
    hypergraphs: list[Hypergraph] = []
    for t in unique_times:
        mask = times == t
        mean_expr = expr[mask].mean(axis=0, keepdims=True).T  # (n_genes, 1)
        hg = from_incidence(
            H_jnp,
            node_features=jnp.array(mean_expr),
        )
        hypergraphs.append(hg)

    return hypergraphs
