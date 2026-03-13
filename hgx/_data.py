"""C. elegans data loaders for hypergraph neural networks.

Provides loaders that fetch and convert C. elegans developmental biology
datasets into :class:`~hgx.Hypergraph` objects.  Downloads are cached
under ``~/.cache/hgx/`` so subsequent calls are instant.

No heavyweight dependencies are required -- only the standard library plus
JAX and NumPy.  All HTTP fetching uses :mod:`urllib.request`.
"""

from __future__ import annotations

import csv
import logging
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hgx"

# Connectome: White et al. 1986 whole-animal, from OpenWorm/c302.
_CONNECTOME_URL = (
    "https://raw.githubusercontent.com/openworm/c302/master/"
    "c302/data/aconnectome_white_1986_whole.csv"
)

# DevoGraph: C. elegans raw cell-tracking centroids.
_DEVOGRAPH_URL = (
    "https://raw.githubusercontent.com/DevoLearn/DevoGraph/master/data/CE_raw_data.csv"
)


def _cache_dir(cache_dir: Path | None = None) -> Path:
    """Return (and create) the cache directory."""
    d = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, returning *dest*.

    If *dest* already exists the download is skipped.

    Raises:
        OSError: If the download fails.
    """
    if dest.exists():
        return dest

    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except urllib.error.URLError as exc:
        raise OSError(
            f"Failed to download {url!r}.  Check your internet connection "
            f"or try again later.  Original error: {exc}"
        ) from exc
    return dest


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts.

    Automatically detects the delimiter (comma or tab).
    """
    with open(path, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(fh, dialect=dialect)
        return list(reader)


# ---------------------------------------------------------------------------
# 1. Connectome
# ---------------------------------------------------------------------------

# Neuron type classification by name prefix.
# Sensory neurons, interneurons, and motor neurons in C. elegans
# are conventionally identified by their names.
_SENSORY_PREFIXES = (
    "ADF",
    "ADL",
    "AFD",
    "ALM",
    "ALN",
    "AQR",
    "ASE",
    "ASG",
    "ASH",
    "ASI",
    "ASJ",
    "ASK",
    "AVM",
    "AWA",
    "AWB",
    "AWC",
    "BAG",
    "CEP",
    "FLP",
    "IL1",
    "IL2",
    "OLL",
    "OLQ",
    "PDE",
    "PHA",
    "PHB",
    "PHC",
    "PLM",
    "PLN",
    "PQR",
    "PVM",
    "URB",
    "URX",
    "URY",
)
_MOTOR_PREFIXES = (
    "DA",
    "DB",
    "DD",
    "VA",
    "VB",
    "VC",
    "VD",
    "AS",
    "RMD",
    "RME",
    "RMF",
    "RMG",
    "RMH",
    "SAA",
    "SAB",
    "SIA",
    "SIB",
    "SMB",
    "SMD",
    "URA",
)


def _neuron_group(name: str) -> str:
    """Classify a neuron name into ``"sensory"``, ``"motor"``, or ``"inter"``."""
    for pfx in _SENSORY_PREFIXES:
        if name.startswith(pfx):
            return "sensory"
    for pfx in _MOTOR_PREFIXES:
        if name.startswith(pfx):
            return "motor"
    return "inter"


def load_connectome(
    *,
    mode: str = "pairwise",
    cache_dir: Path | None = None,
) -> Any:
    """Load the *C. elegans* neural connectome as a hypergraph.

    Data source: White et al. 1986 whole-animal connectome via
    `OpenWorm/c302 <https://github.com/openworm/c302>`_.  The CSV
    contains columns ``pre``, ``post``, ``type``, and ``synapses``.

    Args:
        mode: How to build hyperedges.

            * ``"pairwise"`` -- each synaptic connection ``(pre, post)``
              becomes a 2-uniform hyperedge.  Edge features encode the
              synapse count.  Node features are degree vectors.
            * ``"grouped"`` -- neurons are grouped by type (sensory,
              inter, motor) and each group forms one large hyperedge.
              Node features are one-hot identity.
        cache_dir: Directory for cached downloads.  Defaults to
            ``~/.cache/hgx/``.

    Returns:
        A :class:`~hgx.Hypergraph` instance.
    """
    if mode not in ("pairwise", "grouped"):
        raise ValueError(f"Unknown mode {mode!r}.  Expected 'pairwise' or 'grouped'.")

    dest = _cache_dir(cache_dir) / "aconnectome_white_1986_whole.csv"
    _download(_CONNECTOME_URL, dest)
    rows = _read_csv(dest)

    # Collect unique neuron names, preserving encounter order.
    neurons: dict[str, int] = {}
    for row in rows:
        pre = row["pre"].strip()
        post = row["post"].strip()
        neurons.setdefault(pre, len(neurons))
        neurons.setdefault(post, len(neurons))

    n = len(neurons)

    if mode == "pairwise":
        return _connectome_pairwise(rows, neurons, n)
    else:
        return _connectome_grouped(neurons, n)


def _connectome_pairwise(
    rows: list[dict[str, str]],
    neurons: dict[str, int],
    n: int,
) -> Any:
    """Build a 2-uniform hypergraph from pairwise synaptic connections."""
    from hgx._hypergraph import from_incidence

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for row in rows:
        pre = neurons[row["pre"].strip()]
        post = neurons[row["post"].strip()]
        w = float(row["synapses"])
        edges.append((pre, post))
        weights.append(w)

    m = len(edges)
    H = np.zeros((n, m), dtype=np.float32)
    for k, (i, j) in enumerate(edges):
        H[i, k] = 1.0
        H[j, k] = 1.0

    # Node features: degree (number of hyperedges each neuron belongs to).
    degrees = H.sum(axis=1, keepdims=True)  # (n, 1)

    # Edge features: synapse count.
    edge_feats = np.array(weights, dtype=np.float32).reshape(-1, 1)

    return from_incidence(
        jnp.array(H),
        node_features=jnp.array(degrees),
        edge_features=jnp.array(edge_feats),
    )


def _connectome_grouped(
    neurons: dict[str, int],
    n: int,
) -> Any:
    """Build a hypergraph where each neuron-type group is one hyperedge."""
    from hgx._hypergraph import from_incidence

    groups: dict[str, list[int]] = defaultdict(list)
    for name, idx in neurons.items():
        groups[_neuron_group(name)].append(idx)

    group_names = sorted(groups.keys())
    m = len(group_names)

    H = np.zeros((n, m), dtype=np.float32)
    for k, gname in enumerate(group_names):
        for idx in groups[gname]:
            H[idx, k] = 1.0

    # Node features: one-hot identity.
    feats = np.eye(n, dtype=np.float32)

    return from_incidence(
        jnp.array(H),
        node_features=jnp.array(feats),
    )


# ---------------------------------------------------------------------------
# 2. Cell lineage
# ---------------------------------------------------------------------------

# The first several rounds of the C. elegans embryonic cell lineage.
# Each entry is ``(parent, child1, child2)``.  Names follow standard
# nomenclature.  This covers rounds 1-5 (up to ~64-cell stage).
_LINEAGE_DIVISIONS: list[tuple[str, str, str]] = [
    # Round 1: zygote
    ("P0", "AB", "P1"),
    # Round 2
    ("AB", "ABa", "ABp"),
    ("P1", "EMS", "P2"),
    # Round 3
    ("ABa", "ABal", "ABar"),
    ("ABp", "ABpl", "ABpr"),
    ("EMS", "MS", "E"),
    ("P2", "C", "P3"),
    # Round 4
    ("ABal", "ABala", "ABalp"),
    ("ABar", "ABara", "ABarp"),
    ("ABpl", "ABpla", "ABplp"),
    ("ABpr", "ABpra", "ABprp"),
    ("MS", "MSa", "MSp"),
    ("E", "Ea", "Ep"),
    ("C", "Ca", "Cp"),
    ("P3", "D", "P4"),
    # Round 5
    ("ABala", "ABalaa", "ABalap"),
    ("ABalp", "ABalpa", "ABalpp"),
    ("ABara", "ABaraa", "ABarap"),
    ("ABarp", "ABarpa", "ABarpp"),
    ("ABpla", "ABplaa", "ABplap"),
    ("ABplp", "ABplpa", "ABplpp"),
    ("ABpra", "ABpraa", "ABprap"),
    ("ABprp", "ABprpa", "ABprpp"),
    ("MSa", "MSaa", "MSap"),
    ("MSp", "MSpa", "MSpp"),
    ("Ea", "Eal", "Ear"),
    ("Ep", "Epl", "Epr"),
    ("Ca", "Caa", "Cap"),
    ("Cp", "Cpa", "Cpp"),
    ("D", "Da", "Dp"),
]


def load_cell_lineage(
    *,
    max_depth: int | None = None,
) -> Any:
    """Load the *C. elegans* embryonic cell lineage as a 3-uniform hypergraph.

    Each cell division ``parent -> (child1, child2)`` is encoded as a
    hyperedge ``{parent, child1, child2}``.  The resulting hypergraph is
    3-uniform: every hyperedge contains exactly three vertices.

    The lineage tree is hardcoded from the canonical *C. elegans*
    nomenclature up to approximately the 64-cell stage (5 division
    rounds, 31 divisions, 63 unique cells).

    Args:
        max_depth: Maximum number of division rounds to include
            (1-indexed).  ``None`` includes all available rounds (5).

    Returns:
        A :class:`~hgx.Hypergraph` with 3-uniform hyperedges.  Node
        features are one-hot identity vectors.
    """
    from hgx._hypergraph import from_incidence

    divisions = _LINEAGE_DIVISIONS
    if max_depth is not None:
        # Round k has 2^(k-1) divisions.  Cumulative count up to round k
        # is 2^k - 1.
        cutoff = 2**max_depth - 1
        divisions = divisions[:cutoff]

    # Collect unique cell names (preserving encounter order).
    cells: dict[str, int] = {}
    for parent, c1, c2 in divisions:
        cells.setdefault(parent, len(cells))
        cells.setdefault(c1, len(cells))
        cells.setdefault(c2, len(cells))

    n = len(cells)
    m = len(divisions)

    H = np.zeros((n, m), dtype=np.float32)
    for k, (parent, c1, c2) in enumerate(divisions):
        H[cells[parent], k] = 1.0
        H[cells[c1], k] = 1.0
        H[cells[c2], k] = 1.0

    feats = np.eye(n, dtype=np.float32)

    return from_incidence(
        jnp.array(H),
        node_features=jnp.array(feats),
    )


# ---------------------------------------------------------------------------
# 3. DevoGraph (cell centroids over time)
# ---------------------------------------------------------------------------


def load_devograph(
    *,
    time_step: int | None = None,
    k_neighbors: int = 5,
    cache_dir: Path | None = None,
) -> Any:
    """Load DevoGraph 3D cell centroid data as hypergraph(s).

    Data source: `DevoLearn/DevoGraph
    <https://github.com/DevoLearn/DevoGraph>`_ -- *C. elegans* embryonic
    cell positions tracked over 190 time steps.

    Nodes represent cells at a given time step, with spatial coordinates
    ``(x, y, z)`` plus cell ``size`` as node features (4-d).  Positions
    are the raw 3D coordinates.  Hyperedges are constructed via
    *k*-nearest-neighbour grouping: for each cell, a hyperedge connects
    it to its *k* nearest spatial neighbours (including itself).
    Duplicate edges are removed.

    Args:
        time_step: If given, return a single :class:`~hgx.Hypergraph`
            for that time step (1-indexed, range 1--190).  If *None*,
            return a list of hypergraphs, one per time step.
        k_neighbors: Number of nearest neighbours used to form each
            hyperedge (including the cell itself).  Default 5.
        cache_dir: Directory for cached downloads.  Defaults to
            ``~/.cache/hgx/``.

    Returns:
        A single :class:`~hgx.Hypergraph` when *time_step* is given,
        otherwise a ``list[Hypergraph]`` indexed from 0 (time step 1).
    """
    dest = _cache_dir(cache_dir) / "CE_raw_data.csv"
    _download(_DEVOGRAPH_URL, dest)
    rows = _read_csv(dest)

    # Group rows by time step.
    by_time: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        t = int(row["time"])
        by_time[t].append(row)

    if time_step is not None:
        if time_step not in by_time:
            available = sorted(by_time.keys())
            raise ValueError(
                f"time_step={time_step} not found.  "
                f"Available: {available[0]}..{available[-1]}"
            )
        return _devograph_single(by_time[time_step], k_neighbors)

    # Build one hypergraph per time step, in order.
    results = []
    for t in sorted(by_time.keys()):
        results.append(_devograph_single(by_time[t], k_neighbors))
    return results


def _devograph_single(
    rows: list[dict[str, str]],
    k: int,
) -> Any:
    """Build a hypergraph for a single DevoGraph time step."""
    from hgx._hypergraph import from_incidence

    n = len(rows)
    coords = np.zeros((n, 3), dtype=np.float32)
    sizes = np.zeros((n, 1), dtype=np.float32)
    for i, row in enumerate(rows):
        coords[i, 0] = float(row["x"])
        coords[i, 1] = float(row["y"])
        coords[i, 2] = float(row["z"])
        sizes[i, 0] = float(row["size"])

    # Node features: [x, y, z, size].
    feats = np.concatenate([coords, sizes], axis=1)  # (n, 4)

    # Ensure k does not exceed n.
    k_eff = min(k, n)

    # k-nearest-neighbour hyperedges via brute-force pairwise distances.
    dists = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    knn_indices = np.argsort(dists, axis=1)[:, :k_eff]  # (n, k_eff)

    # Deduplicate: sort each row, keep unique tuples.
    edge_set: set[tuple[int, ...]] = set()
    edge_list: list[tuple[int, ...]] = []
    for i in range(n):
        edge = tuple(sorted(knn_indices[i].tolist()))
        if edge not in edge_set:
            edge_set.add(edge)
            edge_list.append(edge)

    m = len(edge_list)
    H = np.zeros((n, m), dtype=np.float32)
    for col, edge in enumerate(edge_list):
        for node in edge:
            H[node, col] = 1.0

    return from_incidence(
        jnp.array(H),
        node_features=jnp.array(feats),
        positions=jnp.array(coords),
        geometry="euclidean",
    )


# ---------------------------------------------------------------------------
# 4. Synthetic: Zachary's karate club lifted to a hypergraph
# ---------------------------------------------------------------------------

# Adjacency list for Zachary's karate club (34 nodes, 0-indexed).
# Upper triangle only; the graph is undirected.
_KARATE_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
]

# Ground-truth community labels (0 = Mr. Hi's faction, 1 = Officer's).
_KARATE_LABELS: list[int] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


def _bron_kerbosch(
    adj: dict[int, set[int]],
    n: int,
) -> list[list[int]]:
    """Find all maximal cliques via Bron-Kerbosch with pivoting."""
    cliques: list[list[int]] = []

    def _recurse(R: set[int], P: set[int], X: set[int]) -> None:
        if not P and not X:
            cliques.append(sorted(R))
            return
        # Pivot: choose u in P|X that maximises |P & adj[u]|.
        pivot = max(P | X, key=lambda u: len(P & adj[u]))
        for v in list(P - adj[pivot]):
            _recurse(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    _recurse(set(), set(range(n)), set())
    return cliques


def load_synthetic_karate() -> Any:
    """Load Zachary's karate club graph lifted to a hypergraph.

    The classic 34-node social network is converted into a hypergraph by
    lifting every maximal clique of size >= 2 to a single hyperedge.
    Pairwise edges that are not part of any larger clique remain as
    2-edges.

    This is a purely synthetic, offline dataset useful for quick tests
    and demonstrations.

    Returns:
        A :class:`~hgx.Hypergraph`.  Node features are 2-d one-hot
        vectors encoding the ground-truth community label (Mr. Hi vs
        Officer).
    """
    from hgx._hypergraph import from_incidence

    n = 34

    # Build adjacency set.
    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in _KARATE_EDGES:
        adj[i].add(j)
        adj[j].add(i)

    # Find all maximal cliques of size >= 2.
    cliques = [c for c in _bron_kerbosch(adj, n) if len(c) >= 2]

    m = len(cliques)
    H = np.zeros((n, m), dtype=np.float32)
    for k, clique in enumerate(cliques):
        for node in clique:
            H[node, k] = 1.0

    # Node features: 2-d one-hot community label.
    feats = np.zeros((n, 2), dtype=np.float32)
    for i, label in enumerate(_KARATE_LABELS):
        feats[i, label] = 1.0

    return from_incidence(
        jnp.array(H),
        node_features=jnp.array(feats),
    )
