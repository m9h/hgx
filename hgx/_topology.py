"""Topological feature extraction for hypergraphs via persistent homology.

Bridges topological data analysis (TDA) with learned hypergraph
representations. Provides both classical persistent homology (via optional
``giotto-tda`` backend) and a fully differentiable topological layer built
on JAX/Equinox that is JIT-compatible.

Mathematical background
-----------------------
Persistent homology tracks topological features (connected components, loops,
cavities) across a *filtration*.  For a hypergraph with weighted edges the
sublevel-set filtration H_t = {e : w(e) <= t} reveals multi-scale topological
structure.  Birth--death pairs (b_i, d_i) form a *persistence diagram*
summarising the topology.

The *Hodge Laplacian* L_k on the clique expansion captures k-dimensional
topology algebraically: nullity(L_k) = beta_k (the k-th Betti number).
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from hgx._hypergraph import Hypergraph
from hgx._transforms import clique_expansion


# ---------------------------------------------------------------------------
# 1. compute_persistence  (requires optional giotto-tda / ripser)
# ---------------------------------------------------------------------------


def compute_persistence(
    hg: Hypergraph,
    filtration: Literal["weight", "degree", "clique"] = "weight",
    max_dim: int = 1,
) -> list[np.ndarray]:
    """Compute persistence diagrams from a hypergraph.

    Parameters
    ----------
    hg : Hypergraph
        Input hypergraph.
    filtration : {"weight", "degree", "clique"}
        * ``"weight"`` -- use edge features (first column) or incidence
          weights as filtration values.
        * ``"degree"`` -- filtration by vertex degree.
        * ``"clique"`` -- apply clique expansion first, then compute standard
          simplicial persistence on the resulting graph.
    max_dim : int
        Maximum homology dimension to compute.

    Returns
    -------
    list[np.ndarray]
        One array per homology dimension, each of shape ``(num_pairs, 2)``
        containing ``(birth, death)`` pairs.
    """
    try:
        from gtda.homology import VietorisRipsPersistence  # noqa: F401
        _backend = "giotto"
    except ImportError:
        try:
            import ripser as _ripser  # noqa: F401
            _backend = "ripser"
        except ImportError:
            raise ImportError(
                "compute_persistence requires either giotto-tda or ripser. "
                "Install with:  pip install giotto-tda   or   pip install ripser"
            ) from None

    H = np.asarray(hg._masked_incidence())

    if filtration == "weight":
        # Use edge features (first column) as edge weight; fall back to
        # incidence column sums (edge degree) when no features are provided.
        if hg.edge_features is not None:
            weights = np.asarray(hg.edge_features[:, 0])
        else:
            weights = H.sum(axis=0)
        # Build a distance matrix between nodes: d(i,j) = max weight of
        # shared hyperedges (0 if not co-members).
        n = H.shape[0]
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0.0)
        for k in range(H.shape[1]):
            members = np.where(H[:, k] > 0)[0]
            w = weights[k]
            for i in members:
                for j in members:
                    if i != j:
                        dist[i, j] = min(dist[i, j], w)
        # Unreachable pairs stay at inf which ripser/giotto handle fine.

    elif filtration == "degree":
        degrees = H.sum(axis=1)
        n = H.shape[0]
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0.0)
        # Pair-wise filtration value = max degree of the two endpoints.
        A = (H @ H.T)
        A = A - np.diag(np.diag(A))
        rows, cols = np.where(A > 0)
        for i, j in zip(rows, cols):
            dist[i, j] = max(degrees[i], degrees[j])

    elif filtration == "clique":
        A = np.asarray(clique_expansion(hg))
        n = A.shape[0]
        # Shortest-path distance via Floyd-Warshall on unweighted graph.
        dist = np.where(A > 0, 1.0, np.inf)
        np.fill_diagonal(dist, 0.0)
        for via in range(n):
            dist = np.minimum(dist, dist[:, via : via + 1] + dist[via : via + 1, :])
    else:
        raise ValueError(f"Unknown filtration: {filtration!r}")

    # Ensure finite upper bound for non-connected components.
    finite_mask = np.isfinite(dist) & (dist > 0)
    max_finite = dist[finite_mask].max() if np.any(finite_mask) else 1.0
    dist = np.where(np.isfinite(dist), dist, max_finite + 1.0)

    if _backend == "ripser":
        import ripser as _ripser
        result = _ripser.ripser(dist, maxdim=max_dim, distance_matrix=True)
        diagrams: list[np.ndarray] = []
        for dim in range(max_dim + 1):
            dgm = result["dgms"][dim]
            # Remove infinite deaths for cleanliness.
            finite_mask = np.isfinite(dgm[:, 1])
            diagrams.append(dgm[finite_mask])
        return diagrams
    else:
        from gtda.homology import VietorisRipsPersistence
        vrp = VietorisRipsPersistence(
            metric="precomputed",
            homology_dimensions=list(range(max_dim + 1)),
            n_jobs=1,
        )
        dgm_gtda = vrp.fit_transform(dist[np.newaxis])[0]
        diagrams = []
        for dim in range(max_dim + 1):
            mask = dgm_gtda[:, 2] == dim
            pairs = dgm_gtda[mask, :2]
            finite_mask = np.isfinite(pairs[:, 1])
            diagrams.append(pairs[finite_mask])
        return diagrams


# ---------------------------------------------------------------------------
# 2. persistence_landscape  (pure numpy)
# ---------------------------------------------------------------------------


def persistence_landscape(
    diagram: np.ndarray,
    num_landscapes: int = 5,
    resolution: int = 100,
) -> np.ndarray:
    """Convert a persistence diagram to persistence landscapes.

    Persistence landscapes are a stable vectorisation of persistence
    diagrams introduced by Bubenik (2015).  The k-th landscape is the
    k-th largest tent function value at each point along a grid.

    Parameters
    ----------
    diagram : np.ndarray
        Array of shape ``(num_pairs, 2)`` with ``(birth, death)`` pairs.
    num_landscapes : int
        Number of landscape functions to return.
    resolution : int
        Number of sample points along the filtration axis.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_landscapes, resolution)``.
    """
    diagram = np.asarray(diagram, dtype=np.float64)
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError("diagram must have shape (num_pairs, 2)")

    if len(diagram) == 0:
        return np.zeros((num_landscapes, resolution), dtype=np.float64)

    b = diagram[:, 0]
    d = diagram[:, 1]

    t_min = float(b.min())
    t_max = float(d.max())
    if t_min == t_max:
        t_max = t_min + 1.0
    grid = np.linspace(t_min, t_max, resolution)

    # Tent function: lambda(t) = min(t - b, d - t) clipped at 0.
    # Shape: (num_pairs, resolution)
    tent = np.maximum(
        np.minimum(grid[np.newaxis, :] - b[:, np.newaxis],
                    d[:, np.newaxis] - grid[np.newaxis, :]),
        0.0,
    )

    # Sort descending along pairs axis at each grid point.
    tent_sorted = np.sort(tent, axis=0)[::-1]  # (num_pairs, resolution)

    # Pad if fewer pairs than requested landscapes.
    if tent_sorted.shape[0] < num_landscapes:
        pad = np.zeros((num_landscapes - tent_sorted.shape[0], resolution))
        tent_sorted = np.concatenate([tent_sorted, pad], axis=0)

    return tent_sorted[:num_landscapes]


# ---------------------------------------------------------------------------
# 3. persistence_image  (pure numpy)
# ---------------------------------------------------------------------------


def persistence_image(
    diagram: np.ndarray,
    resolution: int = 50,
    sigma: float = 0.1,
) -> np.ndarray:
    """Convert a persistence diagram to a persistence image.

    Uses the birth--persistence representation and a Gaussian kernel
    to produce a stable, fixed-size 2-D descriptor (Adams et al., 2017).

    Parameters
    ----------
    diagram : np.ndarray
        Array of shape ``(num_pairs, 2)`` with ``(birth, death)`` pairs.
    resolution : int
        Grid resolution along each axis.
    sigma : float
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Array of shape ``(resolution, resolution)``.
    """
    diagram = np.asarray(diagram, dtype=np.float64)
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError("diagram must have shape (num_pairs, 2)")

    if len(diagram) == 0:
        return np.zeros((resolution, resolution), dtype=np.float64)

    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]

    b_min, b_max = float(births.min()), float(births.max())
    p_min, p_max = 0.0, float(persistences.max())
    if b_min == b_max:
        b_max = b_min + 1.0
    if p_min == p_max:
        p_max = p_min + 1.0

    b_grid = np.linspace(b_min, b_max, resolution)
    p_grid = np.linspace(p_min, p_max, resolution)
    B, P = np.meshgrid(b_grid, p_grid, indexing="ij")  # (res, res)

    img = np.zeros((resolution, resolution), dtype=np.float64)
    for k in range(len(births)):
        weight = persistences[k]  # linear ramp weighting
        gauss = np.exp(
            -((B - births[k]) ** 2 + (P - persistences[k]) ** 2) / (2 * sigma ** 2)
        )
        img += weight * gauss

    return img


# ---------------------------------------------------------------------------
# 4. persistence_features  (convenience wrapper)
# ---------------------------------------------------------------------------


def persistence_features(
    hg: Hypergraph,
    method: Literal["landscape", "image"] = "landscape",
    **kwargs,
) -> np.ndarray:
    """Compute persistence and vectorise into a feature array.

    This is a convenience function that chains :func:`compute_persistence`
    with either :func:`persistence_landscape` or :func:`persistence_image`.

    Parameters
    ----------
    hg : Hypergraph
        Input hypergraph.
    method : {"landscape", "image"}
        Vectorisation method.
    **kwargs
        Forwarded to ``compute_persistence`` (``filtration``, ``max_dim``)
        and the vectoriser (``num_landscapes``, ``resolution``, ``sigma``).

    Returns
    -------
    np.ndarray
        Feature array.  For landscapes the shape is
        ``(max_dim+1, num_landscapes, resolution)``; for images it is
        ``(max_dim+1, resolution, resolution)``.
    """
    cp_kwargs: dict = {}
    for key in ("filtration", "max_dim"):
        if key in kwargs:
            cp_kwargs[key] = kwargs.pop(key)

    diagrams = compute_persistence(hg, **cp_kwargs)

    if method == "landscape":
        vec_fn = persistence_landscape
    elif method == "image":
        vec_fn = persistence_image
    else:
        raise ValueError(f"Unknown method: {method!r}")

    features = np.stack([vec_fn(dgm, **kwargs) for dgm in diagrams])
    return features


# ---------------------------------------------------------------------------
# 5. TopologicalLayer  (differentiable, JIT-compatible, pure JAX)
# ---------------------------------------------------------------------------


class TopologicalLayer(eqx.Module):
    """Differentiable topological layer for hypergraph features.

    Computes persistence-*like* topological summaries in a fully
    differentiable manner using a soft Rips construction via the heat
    kernel.  The **distance-to-measure (DTM)** filtration is inherently
    smooth with respect to point positions, enabling gradient flow.

    The layer:

    1. Computes pairwise squared distances between node features /
       positions.
    2. Forms a soft similarity matrix via the heat kernel
       ``K_t(i,j) = exp(-d(i,j)^2 / t)``.
    3. Derives topological summary statistics from the spectrum of the
       resulting graph Laplacian (eigenvalues encode Betti-number
       information).

    The output is a fixed-size feature vector of length
    ``2 * num_eigvals`` (eigenvalues + spectral gaps).

    Attributes
    ----------
    num_eigvals : int
        Number of smallest eigenvalues to keep.
    bandwidth : float
        Heat kernel bandwidth parameter *t*.
    """

    num_eigvals: int = eqx.field(static=True)
    bandwidth: float

    def __init__(self, num_eigvals: int = 8, bandwidth: float = 1.0):
        self.num_eigvals = num_eigvals
        self.bandwidth = bandwidth

    def __call__(self, features: Float[Array, "n d"]) -> Float[Array, " f"]:
        """Compute topological features from node features.

        Parameters
        ----------
        features : array of shape ``(n, d)``
            Node feature matrix (or positions).

        Returns
        -------
        array of shape ``(2 * num_eigvals,)``
            Topological summary: eigenvalues and spectral gaps.
        """
        # Pairwise squared distances.
        diff = features[:, None, :] - features[None, :, :]  # (n, n, d)
        sq_dist = jnp.sum(diff ** 2, axis=-1)  # (n, n)

        # Soft Rips / heat kernel similarity.
        K = jnp.exp(-sq_dist / self.bandwidth)  # (n, n)

        # Graph Laplacian from the kernel.
        D = jnp.diag(jnp.sum(K, axis=1))
        L = D - K

        # Eigenvalues (sorted ascending).
        eigvals = jnp.linalg.eigvalsh(L)

        # Take the smallest num_eigvals eigenvalues.
        n = features.shape[0]
        k = min(self.num_eigvals, n)
        selected = eigvals[:k]

        # Pad if n < num_eigvals.
        selected = jnp.concatenate(
            [selected, jnp.zeros(self.num_eigvals - k)]
        )

        # Spectral gaps (consecutive differences).
        gaps = jnp.diff(selected, prepend=jnp.array([0.0]))

        return jnp.concatenate([selected, gaps])


# ---------------------------------------------------------------------------
# 6. hodge_laplacians  (pure JAX)
# ---------------------------------------------------------------------------


def hodge_laplacians(
    hg: Hypergraph,
    max_dim: int = 2,
) -> list[Float[Array, ...]]:
    """Compute the k-th Hodge Laplacians from the hypergraph's clique expansion.

    The Hodge Laplacian ``L_k`` is defined as:

        L_k = B_k^T B_k + B_{k+1} B_{k+1}^T

    where ``B_k`` is the k-th boundary operator of the simplicial complex
    obtained from the clique expansion.  The nullity of ``L_k`` equals
    the k-th Betti number (by the Hodge theorem).

    Parameters
    ----------
    hg : Hypergraph
        Input hypergraph.
    max_dim : int
        Maximum dimension of Hodge Laplacian to compute.

    Returns
    -------
    list[jnp.ndarray]
        ``[L_0, L_1, ..., L_{max_dim}]`` as dense matrices.
    """
    A = jnp.array(clique_expansion(hg))
    n = A.shape[0]

    # ---- Extract simplices from clique expansion --------------------------
    # 0-simplices: vertices
    vertices = list(range(n))

    # 1-simplices: edges (upper triangle of adjacency)
    edge_list: list[tuple[int, int]] = []
    A_np = np.asarray(A)
    for i in range(n):
        for j in range(i + 1, n):
            if A_np[i, j] > 0:
                edge_list.append((i, j))

    # Higher simplices: find cliques by checking common neighbourhoods.
    # We build simplices up to dimension max_dim + 1 (so we have B_{max_dim+1}).
    simplices_by_dim: dict[int, list[tuple[int, ...]]] = {
        0: [(v,) for v in vertices],
        1: [tuple(sorted(e)) for e in edge_list],
    }

    adj_sets: list[set[int]] = []
    for i in range(n):
        adj_sets.append({j for j in range(n) if A_np[i, j] > 0 and j != i})

    for dim in range(2, max_dim + 2):
        simplices_by_dim[dim] = []
        if dim - 1 not in simplices_by_dim:
            break
        for simplex in simplices_by_dim[dim - 1]:
            # Candidates are vertices adjacent to *all* vertices in simplex
            # and greater than the largest vertex (to avoid duplicates).
            candidates = set(adj_sets[simplex[0]])
            for v in simplex[1:]:
                candidates = candidates & adj_sets[v]
            for c in sorted(candidates):
                if c > simplex[-1]:
                    new_simplex = tuple(sorted(simplex + (c,)))
                    simplices_by_dim[dim].append(new_simplex)

    # ---- Build boundary operators -----------------------------------------
    def _boundary_matrix(dim: int) -> jnp.ndarray:
        """Build boundary operator B_dim: dim-simplices to (dim-1)-simplices."""
        if dim < 1 or dim not in simplices_by_dim:
            return jnp.zeros((0, 0))

        higher = simplices_by_dim[dim]
        lower = simplices_by_dim[dim - 1]
        if len(higher) == 0 or len(lower) == 0:
            return jnp.zeros((len(lower), len(higher)))

        lower_to_idx = {s: i for i, s in enumerate(lower)}
        B = np.zeros((len(lower), len(higher)), dtype=np.float32)
        for j, sigma in enumerate(higher):
            for face_idx in range(len(sigma)):
                face = sigma[:face_idx] + sigma[face_idx + 1:]
                if face in lower_to_idx:
                    sign = (-1.0) ** face_idx
                    B[lower_to_idx[face], j] = sign
        return jnp.array(B)

    # ---- Assemble Hodge Laplacians ----------------------------------------
    laplacians: list[jnp.ndarray] = []
    for k in range(max_dim + 1):
        num_k = len(simplices_by_dim.get(k, []))

        # B_k: maps k-simplices -> (k-1)-simplices
        B_k = _boundary_matrix(k)

        # B_{k+1}: maps (k+1)-simplices -> k-simplices
        B_kp1 = _boundary_matrix(k + 1)

        # L_k = B_k^T B_k + B_{k+1} B_{k+1}^T
        if B_k.size > 0 and B_k.shape[0] > 0 and B_k.shape[1] > 0:
            down = B_k.T @ B_k
        else:
            down = jnp.zeros((num_k, num_k))

        if B_kp1.size > 0 and B_kp1.shape[0] > 0 and B_kp1.shape[1] > 0:
            up = B_kp1 @ B_kp1.T
        else:
            up = jnp.zeros((num_k, num_k))

        laplacians.append(down + up)

    return laplacians
