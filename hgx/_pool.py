"""Hierarchical coarsening and pooling operators for hypergraphs.

Provides learned and non-learned pooling methods that coarsen a hypergraph
by clustering nodes into supernodes, producing a smaller hypergraph suitable
for graph-level prediction tasks and multi-resolution analysis.

Mathematical background:
    Given assignment matrix S in R^{n x n'} (n' < n), the coarsened
    incidence is H' = S^T H, coarsened features are X' = S^T X.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._unigcn import UniGCNConv
from hgx._hypergraph import Hypergraph
from hgx._transforms import hypergraph_laplacian


# ---------------------------------------------------------------------------
# Global pooling (no learning)
# ---------------------------------------------------------------------------


def hypergraph_global_pool(
    hg: Hypergraph,
    method: str = "mean",
) -> Float[Array, " d"]:
    """Aggregate all node features into a single graph-level vector.

    Supports ``"mean"``, ``"max"``, and ``"sum"`` aggregation.  When a
    ``node_mask`` is present only active nodes contribute.

    Args:
        hg: Input hypergraph with node features of shape ``(n, d)``.
        method: Aggregation method — one of ``"mean"``, ``"max"``, ``"sum"``.

    Returns:
        Graph-level feature vector of shape ``(d,)``.
    """
    x = hg.node_features  # (n, d)

    if hg.node_mask is not None:
        mask = hg.node_mask[:, None]  # (n, 1)
        x = x * mask
    else:
        mask = None

    if method == "sum":
        return jnp.sum(x, axis=0)
    elif method == "mean":
        if mask is not None:
            count = jnp.maximum(jnp.sum(mask), 1.0)
        else:
            count = jnp.array(x.shape[0], dtype=x.dtype)
        return jnp.sum(x, axis=0) / count
    elif method == "max":
        if mask is not None:
            # Replace masked positions with -inf so they don't win
            x = jnp.where(mask > 0, x, -jnp.inf)
        return jnp.max(x, axis=0)
    else:
        raise ValueError(f"Unknown pooling method: {method!r}")


# ---------------------------------------------------------------------------
# Learned soft-assignment pooling (DiffPool-style)
# ---------------------------------------------------------------------------


class HypergraphPooling(eqx.Module):
    """Learned soft assignment pooling for hypergraphs (DiffPool-style).

    Uses a GNN to compute a soft assignment matrix
    ``S = softmax(conv(hg))`` of shape ``(n, num_clusters)``.  The
    coarsened hypergraph has ``num_clusters`` super-nodes with:

    * Incidence ``H' = S^T H``   (num_clusters x m)
    * Features  ``X' = S^T X``   (num_clusters x d)

    Attributes:
        conv: Convolution layer that produces assignment logits.
        num_clusters: Number of clusters (super-nodes) in the output.
    """

    conv: eqx.Module
    num_clusters: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        num_clusters: int,
        conv_cls: type = UniGCNConv,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize HypergraphPooling.

        Args:
            in_dim: Input feature dimension.
            num_clusters: Number of clusters (super-nodes) in output.
            conv_cls: Convolution class used to compute assignment logits.
            key: PRNG key for weight initialization.
        """
        self.num_clusters = num_clusters
        self.conv = conv_cls(in_dim, num_clusters, key=key)

    def __call__(
        self,
        hg: Hypergraph,
    ) -> tuple[Hypergraph, Float[Array, "n num_clusters"]]:
        """Pool the hypergraph via soft assignment.

        Args:
            hg: Input hypergraph with node features of shape ``(n, d)``.

        Returns:
            A tuple ``(coarsened_hg, S)`` where ``coarsened_hg`` has
            ``num_clusters`` nodes and ``S`` is the soft assignment matrix.
        """
        logits = self.conv(hg)  # (n, num_clusters)
        S = jax.nn.softmax(logits, axis=1)  # rows sum to 1

        H = hg._masked_incidence()  # (n, m)
        X = hg.node_features  # (n, d)

        # Coarsen
        H_coarse = S.T @ H  # (num_clusters, m)
        X_coarse = S.T @ X  # (num_clusters, d)

        coarsened = Hypergraph(
            node_features=X_coarse,
            incidence=H_coarse,
        )
        return coarsened, S

    @staticmethod
    def link_pred_loss(
        S: Float[Array, "n k"],
        H: Float[Array, "n m"],
    ) -> Float[Array, ""]:
        """Link prediction regularisation loss.

        Measures ``||S S^T - A||_F`` where ``A = H H^T`` is the adjacency
        implied by the hypergraph incidence.

        Args:
            S: Soft assignment matrix (n, k).
            H: Incidence matrix (n, m).

        Returns:
            Frobenius norm of the residual (scalar).
        """
        A = H @ H.T  # (n, n) — adjacency via clique expansion (with self-loops)
        SS = S @ S.T  # (n, n)
        return jnp.sqrt(jnp.sum((SS - A) ** 2))

    @staticmethod
    def entropy_loss(
        S: Float[Array, "n k"],
    ) -> Float[Array, ""]:
        """Entropy regularisation loss over assignment rows.

        Encourages each node to be assigned crisply to one cluster.

        Args:
            S: Soft assignment matrix (n, k).

        Returns:
            Mean entropy across nodes (scalar, non-negative).
        """
        # Clamp to avoid log(0)
        S_safe = jnp.clip(S, 1e-8, 1.0)
        entropy_per_node = -jnp.sum(S_safe * jnp.log(S_safe), axis=1)
        return jnp.mean(entropy_per_node)


# ---------------------------------------------------------------------------
# Score-based hard pooling (TopK)
# ---------------------------------------------------------------------------


class TopKPooling(eqx.Module):
    """Score-based hard pooling that keeps top-k nodes by learned importance.

    A convolution layer produces node embeddings which are projected to
    scalar scores via a learnable vector ``p``.  The top ``ceil(ratio * n)``
    nodes (by sigmoid score) are kept; the rest are masked out.  This is
    fully JIT-compatible — no dynamic shapes, only masks.

    Attributes:
        conv: Convolution layer that produces node embeddings.
        proj: Learnable projection vector of shape ``(out_dim,)``.
        ratio: Fraction of nodes to keep.
    """

    conv: eqx.Module
    proj: Float[Array, " d"]
    ratio: float = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        ratio: float = 0.5,
        conv_cls: type = UniGCNConv,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize TopKPooling.

        Args:
            in_dim: Input feature dimension.
            ratio: Fraction of nodes to keep (0 < ratio <= 1).
            conv_cls: Convolution class used to compute node embeddings.
            key: PRNG key for weight initialization.
        """
        k1, k2 = jax.random.split(key)
        self.conv = conv_cls(in_dim, in_dim, key=k1)
        self.proj = jax.random.normal(k2, (in_dim,)) * 0.01
        self.ratio = ratio

    def __call__(self, hg: Hypergraph) -> Hypergraph:
        """Pool the hypergraph by keeping top-k scoring nodes.

        Args:
            hg: Input hypergraph.

        Returns:
            A new Hypergraph with ``node_mask`` and ``edge_mask`` updated so
            that only the top-k nodes (and their incident edges) are active.
        """
        n = hg.node_features.shape[0]
        import math
        k = max(1, math.ceil(self.ratio * n))

        # Score each node
        embeddings = self.conv(hg)  # (n, d)
        scores = jax.nn.sigmoid(embeddings @ self.proj)  # (n,)

        # Mask out already-inactive nodes so they never win
        if hg.node_mask is not None:
            scores = jnp.where(hg.node_mask, scores, -jnp.inf)

        # Top-k via argsort (JIT-friendly: fixed output size)
        ranked = jnp.argsort(-scores)  # descending
        # Build boolean mask: True for top-k indices
        top_k_indices = ranked[:k]
        node_mask = jnp.zeros(n, dtype=jnp.bool_).at[top_k_indices].set(True)

        # Gate node features by score so gradients flow
        gated_features = hg.node_features * scores[:, None]

        # Edge mask: keep edges that still have at least one active node
        H = hg.incidence  # (n, m)
        active_membership = H * node_mask[:, None]  # zero out inactive rows
        edge_active_count = jnp.sum(active_membership, axis=0)  # (m,)
        edge_mask = edge_active_count > 0
        if hg.edge_mask is not None:
            edge_mask = edge_mask & hg.edge_mask

        return Hypergraph(
            node_features=gated_features,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=node_mask,
            edge_mask=edge_mask,
            geometry=hg.geometry,
        )


# ---------------------------------------------------------------------------
# Spectral pooling (no learning)
# ---------------------------------------------------------------------------


class SpectralPooling(eqx.Module):
    """Coarsening via spectral clustering on the hypergraph Laplacian.

    Computes eigenvectors of the normalized hypergraph Laplacian and
    applies k-means in the spectral embedding to obtain hard cluster
    assignments.  This is deterministic (no learnable parameters) and
    useful as an initialisation or baseline.

    Attributes:
        num_clusters: Number of clusters (super-nodes).
        num_kmeans_iters: Number of k-means iterations.
    """

    num_clusters: int = eqx.field(static=True)
    num_kmeans_iters: int = eqx.field(static=True)

    def __init__(self, num_clusters: int, num_kmeans_iters: int = 20):
        """Initialize SpectralPooling.

        Args:
            num_clusters: Number of clusters for coarsening.
            num_kmeans_iters: Iterations for k-means clustering.
        """
        self.num_clusters = num_clusters
        self.num_kmeans_iters = num_kmeans_iters

    def __call__(self, hg: Hypergraph) -> tuple[Hypergraph, Float[Array, " n"]]:
        """Coarsen the hypergraph via spectral clustering.

        Args:
            hg: Input hypergraph.

        Returns:
            A tuple ``(coarsened_hg, assignments)`` where ``assignments``
            is an integer array of shape ``(n,)`` giving the cluster id for
            each node, and ``coarsened_hg`` has ``num_clusters`` nodes.
        """
        L = hypergraph_laplacian(hg, normalized=True)  # (n, n)

        # Eigen-decomposition (symmetric)
        eigenvalues, eigenvectors = jnp.linalg.eigh(L)

        # Take the first num_clusters eigenvectors (smallest eigenvalues)
        V = eigenvectors[:, :self.num_clusters]  # (n, num_clusters)

        # Normalise rows for better clustering
        row_norms = jnp.linalg.norm(V, axis=1, keepdims=True)
        V = V / jnp.maximum(row_norms, 1e-8)

        # Simple k-means (deterministic init: first num_clusters rows)
        centroids = V[:self.num_clusters]  # (k, num_clusters)

        def kmeans_step(centroids, _):
            # Assign each point to nearest centroid
            dists = jnp.sum(
                (V[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )  # (n, k)
            assignments = jnp.argmin(dists, axis=1)  # (n,)

            # Update centroids
            one_hot = jax.nn.one_hot(assignments, self.num_clusters)  # (n, k)
            counts = jnp.sum(one_hot, axis=0, keepdims=True).T  # (k, 1)
            new_centroids = (one_hot.T @ V) / jnp.maximum(counts, 1.0)
            return new_centroids, assignments

        centroids, assignments = jax.lax.scan(
            kmeans_step, centroids, None, length=self.num_kmeans_iters,
        )
        # assignments from scan is (num_iters, n) — take last iteration
        assignments = assignments[-1]

        # Build hard assignment matrix S (n, num_clusters)
        S = jax.nn.one_hot(assignments, self.num_clusters)  # (n, k)

        H = hg._masked_incidence()
        X = hg.node_features

        H_coarse = S.T @ H  # (k, m)
        X_coarse = S.T @ X  # (k, d)

        # Normalise features by cluster size
        counts = jnp.sum(S, axis=0, keepdims=True).T  # (k, 1)
        X_coarse = X_coarse / jnp.maximum(counts, 1.0)

        coarsened = Hypergraph(
            node_features=X_coarse,
            incidence=H_coarse,
        )
        return coarsened, assignments


# ---------------------------------------------------------------------------
# Full hierarchical model
# ---------------------------------------------------------------------------


class HierarchicalHGNN(eqx.Module):
    """Hierarchical hypergraph neural network with alternating conv + pool.

    Applies a sequence of ``conv -> pool -> conv -> pool -> ... -> readout``
    steps to produce a graph-level prediction.  The global readout is a
    mean-pool over all remaining nodes after the final pooling layer.

    Attributes:
        conv_layers: List of convolution layers.
        pool_layers: List of ``HypergraphPooling`` layers.
        readout: Linear head producing the final graph-level output.
    """

    conv_layers: list[eqx.Module]
    pool_layers: list[HypergraphPooling]
    readout: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        conv_dims: list[tuple[int, int]],
        pool_clusters: list[int],
        out_dim: int,
        conv_cls: type = UniGCNConv,
        activation: Callable = jax.nn.relu,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize HierarchicalHGNN.

        The number of conv layers must equal ``len(pool_clusters) + 1``
        (one conv after the last pool, or one conv before the first pool
        when there is only one pool).  Alternatively, the same number of
        conv and pool layers is accepted — the final conv is applied after
        the last pool.

        For simplicity, we require ``len(conv_dims) == len(pool_clusters) + 1``:
        ``conv_0 -> pool_0 -> conv_1 -> pool_1 -> ... -> conv_L -> readout``.

        Args:
            conv_dims: ``(in_dim, out_dim)`` pairs for each conv layer.
            pool_clusters: Number of clusters for each pool layer.
            out_dim: Dimension of the final graph-level output.
            conv_cls: Convolution class for both message-passing and pooling.
            activation: Activation function applied after each conv.
            key: PRNG key.
        """
        num_convs = len(conv_dims)
        num_pools = len(pool_clusters)
        if num_convs != num_pools + 1:
            raise ValueError(
                f"Need len(conv_dims) == len(pool_clusters) + 1, "
                f"got {num_convs} and {num_pools}"
            )

        keys = jax.random.split(key, num_convs + num_pools + 1)

        self.conv_layers = [
            conv_cls(in_d, out_d, key=keys[i])
            for i, (in_d, out_d) in enumerate(conv_dims)
        ]

        self.pool_layers = [
            HypergraphPooling(
                in_dim=conv_dims[i][1],
                num_clusters=pool_clusters[i],
                conv_cls=conv_cls,
                key=keys[num_convs + i],
            )
            for i in range(num_pools)
        ]

        last_dim = conv_dims[-1][1]
        self.readout = eqx.nn.Linear(last_dim, out_dim, key=keys[-1])
        self.activation = activation

    def __call__(self, hg: Hypergraph) -> Float[Array, " out_dim"]:
        """Forward pass: conv -> pool -> ... -> global readout.

        Args:
            hg: Input hypergraph.

        Returns:
            Graph-level prediction vector of shape ``(out_dim,)``.
        """
        x = hg.node_features

        for i, (conv, pool) in enumerate(
            zip(self.conv_layers[:-1], self.pool_layers)
        ):
            # Update features
            hg_i = Hypergraph(
                node_features=x,
                incidence=hg.incidence,
                edge_features=hg.edge_features,
                positions=hg.positions,
                node_mask=hg.node_mask,
                edge_mask=hg.edge_mask,
                geometry=hg.geometry,
            )
            x = self.activation(conv(hg_i))

            # Pool
            hg_pool = Hypergraph(
                node_features=x,
                incidence=hg_i.incidence,
                edge_features=hg_i.edge_features,
                node_mask=hg_i.node_mask,
                edge_mask=hg_i.edge_mask,
                geometry=hg_i.geometry,
            )
            hg, S = pool(hg_pool)
            x = hg.node_features

        # Final conv
        last_conv = self.conv_layers[-1]
        hg_final = Hypergraph(
            node_features=x,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry=hg.geometry,
        )
        x = self.activation(last_conv(hg_final))

        # Global mean readout
        graph_emb = jnp.mean(x, axis=0)  # (d,)

        return self.readout(graph_emb)
