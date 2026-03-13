"""Sparse incidence matrix support for large-scale hypergraphs.

Stores hypergraph structure as COO indices and data arrays rather than a
dense incidence matrix, reducing memory from O(n*m) to O(nnz).  All
operations are JIT-compatible and use ``jax.ops.segment_sum`` for
efficient message passing.

Design note: JAX's ``jax.experimental.sparse.BCOO`` is not always a
stable pytree leaf, so we store raw ``indices`` and ``data`` arrays on
the Equinox module and reconstruct BCOO only when needed for specific
operations.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# Sparse message-passing primitives
# ---------------------------------------------------------------------------


def sparse_vertex_to_edge(
    x: Float[Array, "n d"],
    indices: Int[Array, "nnz 2"],
    data: Float[Array, " nnz"],
    shape: tuple[int, int],
) -> Float[Array, "m d"]:
    """Aggregate vertex features to hyperedges using sparse ops.

    Equivalent to ``H.T @ x`` where H is the dense incidence matrix,
    but computed in O(nnz) via ``jax.ops.segment_sum``.

    Args:
        x: Vertex feature matrix of shape (n, d).
        indices: COO index pairs of shape (nnz, 2) with columns
            [vertex_idx, edge_idx].
        data: Incidence values of shape (nnz,).
        shape: Tuple (n, m) giving the logical incidence matrix shape.

    Returns:
        Aggregated hyperedge features of shape (m, d).
    """
    _n, m = shape
    v_idx = indices[:, 0]
    e_idx = indices[:, 1]
    vals = x[v_idx] * data[:, None]
    return jax.ops.segment_sum(vals, e_idx, num_segments=m)


def sparse_edge_to_vertex(
    e: Float[Array, "m d"],
    indices: Int[Array, "nnz 2"],
    data: Float[Array, " nnz"],
    shape: tuple[int, int],
) -> Float[Array, "n d"]:
    """Aggregate hyperedge features to vertices using sparse ops.

    Equivalent to ``H @ e`` where H is the dense incidence matrix,
    but computed in O(nnz) via ``jax.ops.segment_sum``.

    Args:
        e: Hyperedge feature matrix of shape (m, d).
        indices: COO index pairs of shape (nnz, 2) with columns
            [vertex_idx, edge_idx].
        data: Incidence values of shape (nnz,).
        shape: Tuple (n, m) giving the logical incidence matrix shape.

    Returns:
        Aggregated vertex features of shape (n, d).
    """
    n, _m = shape
    v_idx = indices[:, 0]
    e_idx = indices[:, 1]
    vals = e[e_idx] * data[:, None]
    return jax.ops.segment_sum(vals, v_idx, num_segments=n)


# ---------------------------------------------------------------------------
# SparseHypergraph data structure
# ---------------------------------------------------------------------------


class SparseHypergraph(eqx.Module):
    """A hypergraph with sparse incidence representation.

    Stores the incidence structure as COO arrays instead of a dense
    ``(n, m)`` matrix, enabling efficient handling of large-scale
    hypergraphs with millions of nodes and edges.

    Attributes:
        node_features: Feature matrix of shape (num_nodes, node_dim).
        indices: COO index pairs of shape (nnz, 2). Each row is
            ``[vertex_idx, edge_idx]`` indicating that the vertex belongs
            to the hyperedge with the associated incidence value.
        data: Incidence values of shape (nnz,).
        shape: Tuple ``(n, m)`` giving the logical incidence matrix
            dimensions. Marked static so that it is not traced by JAX.
        edge_features: Optional feature matrix of shape (num_edges, edge_dim).
        positions: Optional spatial coordinates of shape (num_nodes, spatial_dim).
        node_mask: Boolean mask of shape (num_nodes,). ``True`` for active nodes.
        edge_mask: Boolean mask of shape (num_edges,). ``True`` for active edges.
        geometry: Coordinate space of positions (static metadata).
    """

    node_features: Float[Array, "n d"]
    indices: Int[Array, "nnz 2"]
    data: Float[Array, " nnz"]
    shape: tuple[int, int] = eqx.field(static=True)
    edge_features: Float[Array, "m de"] | None = None
    positions: Float[Array, "n s"] | None = None
    node_mask: Bool[Array, " n"] | None = None
    edge_mask: Bool[Array, " m"] | None = None
    geometry: str | None = eqx.field(static=True, default=None)

    # -- Properties ---------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of (active) nodes."""
        if self.node_mask is not None:
            return int(jnp.sum(self.node_mask))
        return self.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of (active) hyperedges."""
        if self.edge_mask is not None:
            return int(jnp.sum(self.edge_mask))
        return self.shape[1]

    @property
    def nnz(self) -> int:
        """Number of non-zero entries in the incidence matrix."""
        return self.indices.shape[0]

    # -- Masking ------------------------------------------------------------

    def _masked_incidence_sparse(
        self,
    ) -> tuple[Int[Array, "nnz 2"], Float[Array, " nnz"]]:
        """Return indices and data with node/edge masks applied.

        Masked entries have their ``data`` zeroed out so that they do
        not contribute to segment-sum aggregation, while the array
        shapes remain fixed for JIT compatibility.

        Returns:
            Tuple ``(indices, data)`` with masks applied via zeroing.
        """
        masked_data = self.data
        if self.node_mask is not None:
            v_idx = self.indices[:, 0]
            masked_data = masked_data * self.node_mask[v_idx].astype(masked_data.dtype)
        if self.edge_mask is not None:
            e_idx = self.indices[:, 1]
            masked_data = masked_data * self.edge_mask[e_idx].astype(masked_data.dtype)
        return self.indices, masked_data

    # -- Conversion ---------------------------------------------------------

    def to_dense(self) -> Hypergraph:
        """Convert this sparse hypergraph to a dense ``Hypergraph``.

        Returns:
            A ``Hypergraph`` with a full dense incidence matrix.
        """
        n, m = self.shape
        H = jnp.zeros((n, m)).at[self.indices[:, 0], self.indices[:, 1]].set(
            self.data
        )
        return Hypergraph(
            node_features=self.node_features,
            incidence=H,
            edge_features=self.edge_features,
            positions=self.positions,
            node_mask=self.node_mask,
            edge_mask=self.edge_mask,
            geometry=self.geometry,
        )


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------


def from_sparse_incidence(
    node_features: Float[Array, "n d"],
    indices: Int[Array, "nnz 2"],
    data: Float[Array, " nnz"],
    shape: tuple[int, int],
    *,
    edge_features: Float[Array, "m de"] | None = None,
    positions: Float[Array, "n s"] | None = None,
    node_mask: Bool[Array, " n"] | None = None,
    edge_mask: Bool[Array, " m"] | None = None,
    geometry: str | None = None,
) -> SparseHypergraph:
    """Create a SparseHypergraph from COO incidence data.

    Args:
        node_features: Node feature matrix of shape (n, d).
        indices: COO index pairs of shape (nnz, 2), each row
            ``[vertex_idx, edge_idx]``.
        data: Incidence values of shape (nnz,).
        shape: Logical incidence matrix shape ``(n, m)``.
        edge_features: Optional hyperedge features.
        positions: Optional spatial coordinates for nodes.
        node_mask: Optional boolean mask for nodes.
        edge_mask: Optional boolean mask for edges.
        geometry: Coordinate space of positions.

    Returns:
        A ``SparseHypergraph`` instance.
    """
    return SparseHypergraph(
        node_features=node_features,
        indices=jnp.asarray(indices, dtype=jnp.int32),
        data=jnp.asarray(data, dtype=jnp.float32),
        shape=shape,
        edge_features=edge_features,
        positions=positions,
        node_mask=node_mask,
        edge_mask=edge_mask,
        geometry=geometry,
    )


def to_sparse(hg: Hypergraph) -> SparseHypergraph:
    """Convert a dense ``Hypergraph`` to a ``SparseHypergraph``.

    Extracts non-zero entries from the dense incidence matrix and
    stores them as COO arrays.

    Args:
        hg: A dense ``Hypergraph``.

    Returns:
        An equivalent ``SparseHypergraph``.
    """
    H = hg.incidence
    n, m = H.shape
    nnz_count = int(jnp.sum(H != 0))
    v_idx, e_idx = jnp.nonzero(H, size=nnz_count, fill_value=0)
    indices = jnp.stack([v_idx, e_idx], axis=1).astype(jnp.int32)
    data = H[v_idx, e_idx].astype(jnp.float32)
    return SparseHypergraph(
        node_features=hg.node_features,
        indices=indices,
        data=data,
        shape=(n, m),
        edge_features=hg.edge_features,
        positions=hg.positions,
        node_mask=hg.node_mask,
        edge_mask=hg.edge_mask,
        geometry=hg.geometry,
    )


def from_edge_list_sparse(
    edge_list: list[set[int] | list[int] | tuple[int, ...]],
    node_features: Float[Array, "n d"] | None = None,
    num_nodes: int | None = None,
    *,
    edge_features: Float[Array, "m de"] | None = None,
    positions: Float[Array, "n s"] | None = None,
    geometry: str | None = None,
) -> SparseHypergraph:
    """Build a ``SparseHypergraph`` directly from an edge list.

    Constructs the COO representation without ever materialising the
    full dense incidence matrix, keeping memory at O(nnz).

    Args:
        edge_list: List of hyperedges. Each element is a set, list, or
            tuple of vertex indices belonging to that hyperedge.
        node_features: Optional node feature matrix of shape (n, d).
            Defaults to ones of dim 1.
        num_nodes: Total number of nodes. Inferred from ``edge_list``
            if not provided.
        edge_features: Optional hyperedge features.
        positions: Optional spatial coordinates for nodes.
        geometry: Coordinate space of positions.

    Returns:
        A ``SparseHypergraph`` instance.
    """
    all_v: list[int] = []
    all_e: list[int] = []
    for k, edge in enumerate(edge_list):
        for v in sorted(edge):
            all_v.append(v)
            all_e.append(k)

    if num_nodes is None:
        num_nodes = max(all_v) + 1 if all_v else 0
    num_edges = len(edge_list)

    if node_features is None:
        node_features = jnp.ones((num_nodes, 1))

    indices = jnp.stack(
        [jnp.array(all_v, dtype=jnp.int32), jnp.array(all_e, dtype=jnp.int32)],
        axis=1,
    )
    data = jnp.ones(len(all_v), dtype=jnp.float32)

    return SparseHypergraph(
        node_features=node_features,
        indices=indices,
        data=data,
        shape=(num_nodes, num_edges),
        edge_features=edge_features,
        positions=positions,
        geometry=geometry,
    )


# ---------------------------------------------------------------------------
# SparseUniGCNConv
# ---------------------------------------------------------------------------


class SparseUniGCNConv(eqx.Module):
    """UniGCN convolution that operates natively on ``SparseHypergraph``.

    Accepts either a ``SparseHypergraph`` or a dense ``Hypergraph``
    (which is converted internally). Uses ``segment_sum``-based
    aggregation for O(nnz) message passing.

    Attributes:
        linear: Linear projection applied to node features.
        normalize: Whether to apply degree normalisation.
    """

    linear: eqx.nn.Linear
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        normalize: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.linear = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=key)
        self.normalize = normalize

    def __call__(
        self, hg: SparseHypergraph | Hypergraph
    ) -> Float[Array, "n out_dim"]:
        """Apply UniGCN convolution.

        Args:
            hg: Input hypergraph (sparse or dense).

        Returns:
            Updated node features of shape (num_nodes, out_dim).
        """
        # Convert dense to sparse if needed
        if isinstance(hg, Hypergraph):
            hg = to_sparse(hg)

        indices, masked_data = hg._masked_incidence_sparse()
        n, m = hg.shape

        x = jax.vmap(self.linear)(hg.node_features)

        if self.normalize:
            # Compute edge degrees: d_e[k] = sum of data for edge k
            d_e = jax.ops.segment_sum(masked_data, indices[:, 1], num_segments=m)
            # Compute vertex degrees: d_v[i] = sum of data for vertex i
            d_v = jax.ops.segment_sum(masked_data, indices[:, 0], num_segments=n)

            d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
            d_v_inv = jnp.where(d_v > 0, 1.0 / d_v, 0.0)

            # Stage 1: V -> E with edge-degree normalisation
            e = sparse_vertex_to_edge(x, indices, masked_data, (n, m))
            e = e * d_e_inv[:, None]

            # Stage 2: E -> V with vertex-degree normalisation
            out = sparse_edge_to_vertex(e, indices, masked_data, (n, m))
            out = out * d_v_inv[:, None]
        else:
            e = sparse_vertex_to_edge(x, indices, masked_data, (n, m))
            out = sparse_edge_to_vertex(e, indices, masked_data, (n, m))

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
