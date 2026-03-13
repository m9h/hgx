"""Optimal transport utilities for hypergraphs.

Provides Sinkhorn-based optimal transport primitives and OT-enhanced
message-passing layers for hypergraph neural networks. All implementations
are pure JAX -- no external OT libraries required -- and are fully
differentiable, JIT-compatible, and GPU-friendly.

Mathematical background
-----------------------
Optimal transport defines principled distances between probability
distributions. The Sinkhorn algorithm solves the entropic-regularised
optimal transport problem:

    min_{T >= 0}  <C, T> + epsilon * KL(T || a b^T)
    s.t.  T 1 = a,  T^T 1 = b

in a differentiable manner via matrix scaling. This module builds on
Sinkhorn to provide:

* Wasserstein distances between point clouds,
* free-support Wasserstein barycenters for aggregation,
* Gromov-Wasserstein distances for comparing metric spaces without
  node correspondence,
* fused Gromov-Wasserstein for joint feature + structure alignment,
* an OT-based convolution layer (``OTConv``) that replaces mean/sum
  aggregation with barycenter aggregation, and
* an ``OTLayer`` for differentiable soft alignment between two
  hypergraphs.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph
from hgx._transforms import hypergraph_laplacian


# ---------------------------------------------------------------------------
# 1. Sinkhorn algorithm (log-domain stabilised)
# ---------------------------------------------------------------------------

def _sinkhorn_core(
    C: Float[Array, "n m"],
    a: Float[Array, " n"],
    b: Float[Array, " m"],
    epsilon: float = 0.1,
    max_iters: int = 100,
) -> Float[Array, "n m"]:
    """Core log-domain Sinkhorn returning only the transport plan."""
    log_a = jnp.log(jnp.maximum(a, 1e-30))
    log_b = jnp.log(jnp.maximum(b, 1e-30))
    log_K = -C / epsilon  # (n, m)

    f = jnp.zeros_like(a)  # (n,)
    g = jnp.zeros_like(b)  # (m,)

    def body(carry, _):
        f, g = carry
        f = log_a - jax.nn.logsumexp(
            log_K + g[None, :], axis=1
        )
        g = log_b - jax.nn.logsumexp(
            log_K + f[:, None], axis=0
        )
        return (f, g), None

    (f, g), _ = jax.lax.scan(
        body, (f, g), None, length=max_iters
    )

    log_T = f[:, None] + log_K + g[None, :]
    P = jnp.exp(log_T)
    return P


def sinkhorn(
    C: Float[Array, "n m"],
    a: Float[Array, " n"],
    b: Float[Array, " m"],
    epsilon: float = 0.1,
    max_iters: int = 100,
) -> Float[Array, "n m"]:
    """Entropy-regularised OT via Sinkhorn iterations.

    Args:
        C: Cost matrix ``(n, m)``.
        a: Source marginal ``(n,)``.
        b: Target marginal ``(m,)``.
        epsilon: Regularisation strength.
        max_iters: Iteration count.

    Returns:
        Transport plan ``P`` of shape ``(n, m)``.
    """
    return _sinkhorn_core(
        C, a, b, epsilon=epsilon, max_iters=max_iters
    )


def _sinkhorn_with_cost(
    a: Float[Array, " n"],
    b: Float[Array, " m"],
    cost: Float[Array, "n m"],
    epsilon: float = 0.1,
    num_iters: int = 50,
) -> tuple[Float[Array, "n m"], float]:
    """Internal helper returning ``(T, cost_val)``."""
    T = _sinkhorn_core(
        cost, a, b, epsilon=epsilon, max_iters=num_iters
    )
    cost_val = jnp.sum(T * cost)
    return T, cost_val


# ---------------------------------------------------------------------------
# Unbalanced Sinkhorn
# ---------------------------------------------------------------------------


def unbalanced_sinkhorn(
    a: Float[Array, " n"],
    b: Float[Array, " m"],
    cost: Float[Array, "n m"],
    epsilon: float = 0.1,
    tau: float = 1.0,
    num_iters: int = 50,
) -> tuple[Float[Array, "n m"], float]:
    """Unbalanced Sinkhorn algorithm with KL marginal penalties.

    Solves the unbalanced entropic OT problem:

        min_T  <C, T> + epsilon * KL(T || a b^T)
                      + tau * KL(T 1 || a)
                      + tau * KL(T^T 1 || b)

    Args:
        a: Source distribution ``(n,)``.
        b: Target distribution ``(m,)``.
        cost: Cost matrix ``(n, m)``.
        epsilon: Entropic regularisation.
        tau: KL penalty weight on marginal violations.
        num_iters: Number of Sinkhorn iterations.

    Returns:
        Tuple ``(T, cost_val)`` -- transport plan and cost.
    """
    rho = tau
    lam = rho / (rho + epsilon)

    log_a = jnp.log(jnp.maximum(a, 1e-30))
    log_b = jnp.log(jnp.maximum(b, 1e-30))
    log_K = -cost / epsilon

    f = jnp.zeros_like(a)
    g = jnp.zeros_like(b)

    def body(carry, _):
        f, g = carry
        f_new = lam * log_a - lam * jax.nn.logsumexp(
            log_K + g[None, :], axis=1
        )
        g_new = lam * log_b - lam * jax.nn.logsumexp(
            log_K + f_new[:, None], axis=0
        )
        return (f_new, g_new), None

    (f, g), _ = jax.lax.scan(
        body, (f, g), None, length=num_iters
    )

    log_T = f[:, None] + log_K + g[None, :]
    T = jnp.exp(log_T)
    cost_val = jnp.sum(T * cost)
    return T, cost_val


# ---------------------------------------------------------------------------
# 2. Cost matrices
# ---------------------------------------------------------------------------


def feature_cost_matrix(
    X: Float[Array, "n d"],
    Y: Float[Array, "m d"],
    metric: str = "euclidean",
) -> Float[Array, "n m"]:
    """Pairwise distances between two feature matrices.

    Args:
        X: Source features of shape ``(n, d)``.
        Y: Target features of shape ``(m, d)``.
        metric: ``"euclidean"`` (default), ``"sqeuclidean"``,
            or ``"cosine"``.

    Returns:
        Cost matrix of shape ``(n, m)``.
    """
    diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
    sq_dist = jnp.sum(diff**2, axis=-1)  # (n, m)

    if metric == "sqeuclidean":
        return sq_dist
    elif metric == "euclidean":
        return jnp.sqrt(jnp.maximum(sq_dist, 1e-30))
    elif metric == "cosine":
        x_norm = jnp.sqrt(
            jnp.maximum(
                jnp.sum(X**2, axis=-1, keepdims=True),
                1e-30,
            )
        )
        y_norm = jnp.sqrt(
            jnp.maximum(
                jnp.sum(Y**2, axis=-1, keepdims=True),
                1e-30,
            )
        )
        similarity = (X / x_norm) @ (Y / y_norm).T
        return 1.0 - similarity
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Expected "
            "'euclidean', 'sqeuclidean', or 'cosine'."
        )


def structural_cost_matrix(
    hg1: Hypergraph,
    hg2: Hypergraph,
) -> Float[Array, "n1 n2"]:
    """Cost matrix based on Laplacian spectra comparison.

    Computes the top-k eigenvectors of each normalised Laplacian
    to form per-node spectral signatures, then returns pairwise
    Euclidean distances between the signatures.

    Args:
        hg1: First hypergraph.
        hg2: Second hypergraph.

    Returns:
        Cost matrix of shape ``(n1, n2)``.
    """
    L1 = hypergraph_laplacian(hg1, normalized=True)
    L2 = hypergraph_laplacian(hg2, normalized=True)
    n1 = L1.shape[0]
    n2 = L2.shape[0]

    k = min(n1, n2, 8)

    _, evecs1 = jnp.linalg.eigh(L1)
    _, evecs2 = jnp.linalg.eigh(L2)

    spec1 = evecs1[:, :k]  # (n1, k)
    spec2 = evecs2[:, :k]  # (n2, k)

    diff = spec1[:, None, :] - spec2[None, :, :]
    cost = jnp.sqrt(
        jnp.maximum(jnp.sum(diff**2, axis=-1), 1e-30)
    )
    return cost


# ---------------------------------------------------------------------------
# 3. Wasserstein distance between point clouds
# ---------------------------------------------------------------------------


def wasserstein_distance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    p: int = 2,
    epsilon: float = 0.1,
) -> float:
    """Wasserstein-p distance between two point clouds.

    Args:
        x: Source point cloud ``(n, d)``.
        y: Target point cloud ``(m, d)``.
        p: Order of the Wasserstein distance (1 or 2).
        epsilon: Sinkhorn regularisation.

    Returns:
        Scalar Wasserstein-p distance.
    """
    n = x.shape[0]
    m = y.shape[0]

    diff = x[:, None, :] - y[None, :, :]
    dist = jnp.sum(diff**2, axis=-1)

    if p == 1:
        cost = jnp.sqrt(jnp.maximum(dist, 1e-30))
    elif p == 2:
        cost = dist
    else:
        cost = jnp.power(
            jnp.maximum(dist, 1e-30), p / 2.0
        )

    a = jnp.ones(n) / n
    b = jnp.ones(m) / m

    _, ot_cost = _sinkhorn_with_cost(
        a, b, cost, epsilon=epsilon
    )

    if p == 2:
        return jnp.sqrt(jnp.maximum(ot_cost, 0.0))
    elif p == 1:
        return ot_cost
    else:
        return jnp.power(
            jnp.maximum(ot_cost, 0.0), 1.0 / p
        )


# ---------------------------------------------------------------------------
# 4. Wasserstein barycenter (free-support, iterative Sinkhorn)
# ---------------------------------------------------------------------------


def wasserstein_barycenter(
    distributions: list[Float[Array, "n_i d"]],
    weights: Float[Array, " K"],
    support_size: int,
    epsilon: float = 0.1,
    num_iters: int = 100,
) -> Float[Array, "support_size d"]:
    """Free-support Wasserstein barycenter via iterative Sinkhorn.

    Given *K* point-cloud distributions, computes a barycenter
    point cloud that minimises the weighted sum of Wasserstein
    distances.

    Args:
        distributions: List of *K* point clouds ``(n_i, d)``.
        weights: Importance weight per distribution ``(K,)``.
        support_size: Number of barycenter support points.
        epsilon: Sinkhorn regularisation.
        num_iters: Number of outer (barycenter) iterations.

    Returns:
        Barycenter point cloud ``(support_size, d)``.
    """
    K = len(distributions)
    weights = weights / jnp.sum(weights)

    all_points = jnp.concatenate(distributions, axis=0)
    n_total = all_points.shape[0]
    indices = jnp.linspace(
        0, n_total - 1, support_size
    ).astype(jnp.int32)
    barycenter = all_points[indices]

    b = jnp.ones(support_size) / support_size

    def outer_step(barycenter, _):
        new_bary = jnp.zeros_like(barycenter)
        for k in range(K):
            pts_k = distributions[k]
            n_k = pts_k.shape[0]
            a_k = jnp.ones(n_k) / n_k

            diff = (
                pts_k[:, None, :]
                - barycenter[None, :, :]
            )
            cost_k = jnp.sum(diff**2, axis=-1)

            T_k, _ = _sinkhorn_with_cost(
                a_k, b, cost_k,
                epsilon=epsilon, num_iters=20,
            )

            col_sums = jnp.maximum(
                jnp.sum(T_k, axis=0, keepdims=True),
                1e-30,
            )
            T_norm = T_k / col_sums
            update = T_norm.T @ pts_k
            new_bary = new_bary + weights[k] * update

        return new_bary, None

    barycenter, _ = jax.lax.scan(
        outer_step, barycenter, None, length=num_iters
    )
    return barycenter


# ---------------------------------------------------------------------------
# 5. OT-based hyperedge aggregation (V -> E)
# ---------------------------------------------------------------------------


def ot_hyperedge_aggregation(
    node_features: Float[Array, "n d"],
    incidence: Float[Array, "n m"],
    epsilon: float = 0.1,
) -> Float[Array, "m d"]:
    """OT-based vertex-to-edge message passing.

    For each hyperedge, computes the OT-weighted mean of member
    node features via a single Sinkhorn solve.

    Args:
        node_features: Node feature matrix ``(n, d)``.
        incidence: Binary incidence matrix ``(n, m)``.
        epsilon: Sinkhorn regularisation.

    Returns:
        Edge features ``(m, d)``.
    """
    def aggregate_one_edge(edge_col):
        membership = edge_col
        degree = jnp.sum(membership)
        a = membership / jnp.maximum(degree, 1e-30)
        b_single = jnp.ones(1)

        candidate = jnp.sum(
            node_features * membership[:, None],
            axis=0,
            keepdims=True,
        )
        diff = node_features - candidate
        cost = jnp.sum(
            diff**2, axis=-1, keepdims=True
        )

        T, _ = _sinkhorn_with_cost(
            a, b_single, cost,
            epsilon=epsilon, num_iters=10,
        )
        agg = T[:, 0] @ node_features
        return agg

    edge_features = jax.vmap(
        aggregate_one_edge, in_axes=1
    )(incidence)
    return edge_features


# ---------------------------------------------------------------------------
# 6. Gromov-Wasserstein distance
# ---------------------------------------------------------------------------


def gromov_wasserstein(
    C1: Float[Array, "n n"],
    C2: Float[Array, "m m"],
    p: Float[Array, " n"],
    q: Float[Array, " m"],
    epsilon: float = 0.1,
    max_iters: int = 50,
) -> tuple[Float[Array, "n m"], float]:
    """Gromov-Wasserstein distance for comparing metric spaces.

    Solves the GW problem via alternating Sinkhorn projections:

        min_T  sum |C1[i,i'] - C2[j,j']|^2 T[i,j] T[i',j']
        s.t.  T 1 = p,  T^T 1 = q

    Uses efficient tensor factorisation of the GW cost.

    Args:
        C1: Intra-distance matrix for space 1 ``(n, n)``.
        C2: Intra-distance matrix for space 2 ``(m, m)``.
        p: Source marginal ``(n,)``.  Must sum to 1.
        q: Target marginal ``(m,)``.  Must sum to 1.
        epsilon: Sinkhorn regularisation.
        max_iters: Number of outer GW iterations.

    Returns:
        Tuple ``(T, gw_cost)`` -- transport plan ``(n, m)``
        and scalar Gromov-Wasserstein cost.
    """
    n = C1.shape[0]
    m = C2.shape[0]

    C1_sq = C1**2
    C2_sq = C2**2

    def compute_gw_cost(T):
        term1 = C1_sq @ T @ jnp.ones((m, 1))  # (n, 1)
        term2 = jnp.ones((1, n)) @ T @ C2_sq  # (1, m)
        term3 = C1 @ T @ C2.T  # (n, m)
        return term1 + term2 - 2 * term3  # (n, m)

    T = p[:, None] * q[None, :]

    def body(carry, _):
        T = carry
        cost = compute_gw_cost(T)
        T_new = _sinkhorn_core(
            cost, p, q, epsilon=epsilon, max_iters=10
        )
        return T_new, None

    T, _ = jax.lax.scan(body, T, None, length=max_iters)

    cost_final = compute_gw_cost(T)
    gw_cost = jnp.sum(T * cost_final)
    return T, gw_cost


# ---------------------------------------------------------------------------
# 7. Hypergraph-specific convenience functions
# ---------------------------------------------------------------------------


def hypergraph_wasserstein(
    hg1: Hypergraph,
    hg2: Hypergraph,
    epsilon: float = 0.1,
) -> float:
    """Wasserstein distance between hypergraphs via node features.

    Treats node features as point-cloud distributions with
    uniform weights and computes the entropic Wasserstein-2
    distance.

    Args:
        hg1: First hypergraph.
        hg2: Second hypergraph.
        epsilon: Sinkhorn regularisation.

    Returns:
        Scalar Wasserstein distance.
    """
    return wasserstein_distance(
        hg1.node_features, hg2.node_features,
        p=2, epsilon=epsilon,
    )


def hypergraph_gromov_wasserstein(
    hg1: Hypergraph,
    hg2: Hypergraph,
    epsilon: float = 0.1,
) -> tuple[Float[Array, "n1 n2"], float]:
    """Gromov-Wasserstein distance between two hypergraphs.

    Uses Laplacian-based intra-distance matrices and uniform
    marginals.

    Args:
        hg1: First hypergraph.
        hg2: Second hypergraph.
        epsilon: Sinkhorn regularisation.

    Returns:
        Tuple ``(T, gw_dist)`` -- transport plan and GW dist.
    """
    L1 = hypergraph_laplacian(hg1, normalized=True)
    L2 = hypergraph_laplacian(hg2, normalized=True)
    n1 = L1.shape[0]
    n2 = L2.shape[0]

    p = jnp.ones(n1) / n1
    q = jnp.ones(n2) / n2

    return gromov_wasserstein(
        L1, L2, p, q, epsilon=epsilon, max_iters=50
    )


def ot_hypergraph_alignment(
    hg1: Hypergraph,
    hg2: Hypergraph,
    alpha: float = 0.5,
    epsilon: float = 0.1,
) -> tuple[Float[Array, "n1 n2"], float]:
    """Fused Gromov-Wasserstein alignment of two hypergraphs.

    Interpolates between feature and structure costs:

        cost = alpha * feature_cost + (1 - alpha) * gw_cost

    Args:
        hg1: First hypergraph.
        hg2: Second hypergraph.
        alpha: Interpolation weight in ``[0, 1]``.
        epsilon: Sinkhorn regularisation.

    Returns:
        Tuple ``(T, fgw_cost)`` -- transport plan ``(n1, n2)``
        and scalar fused GW cost.
    """
    L1 = hypergraph_laplacian(hg1, normalized=True)
    L2 = hypergraph_laplacian(hg2, normalized=True)
    n1 = L1.shape[0]
    n2 = L2.shape[0]

    p = jnp.ones(n1) / n1
    q = jnp.ones(n2) / n2

    feat_cost = feature_cost_matrix(
        hg1.node_features, hg2.node_features,
        metric="sqeuclidean",
    )

    C1_sq = L1**2
    C2_sq = L2**2

    def compute_gw_cost(T):
        term1 = C1_sq @ T @ jnp.ones((n2, 1))  # (n1, 1)
        term2 = jnp.ones((1, n1)) @ T @ C2_sq  # (1, n2)
        term3 = L1 @ T @ L2.T  # (n1, n2)
        return term1 + term2 - 2 * term3  # (n1, n2)

    T = p[:, None] * q[None, :]

    def body(carry, _):
        T = carry
        gw_cost_mat = compute_gw_cost(T)
        fused = (
            alpha * feat_cost
            + (1.0 - alpha) * gw_cost_mat
        )
        T_new = _sinkhorn_core(
            fused, p, q,
            epsilon=epsilon, max_iters=10,
        )
        return T_new, None

    T, _ = jax.lax.scan(body, T, None, length=50)

    gw_cost_final = compute_gw_cost(T)
    fgw_cost = jnp.sum(
        T * (
            alpha * feat_cost
            + (1.0 - alpha) * gw_cost_final
        )
    )
    return T, fgw_cost


# ---------------------------------------------------------------------------
# 8. OTLayer -- differentiable alignment layer
# ---------------------------------------------------------------------------


class OTLayer(eqx.Module):
    """Differentiable OT layer for soft alignment of hypergraphs.

    Computes an optimal transport plan between node feature
    distributions of two hypergraphs and returns both the plan
    and features from ``hg1`` aligned to ``hg2``'s space.

    Attributes:
        epsilon: Sinkhorn regularisation parameter.
        max_iters: Number of Sinkhorn iterations.
    """

    epsilon: float = eqx.field(static=True)
    max_iters: int = eqx.field(static=True)

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iters: int = 100,
    ):
        """Initialise OTLayer.

        Args:
            epsilon: Sinkhorn regularisation strength.
            max_iters: Number of Sinkhorn iterations.
        """
        self.epsilon = epsilon
        self.max_iters = max_iters

    def __call__(
        self,
        hg1: Hypergraph,
        hg2: Hypergraph,
    ) -> tuple[
        Float[Array, "n1 n2"],
        Float[Array, "n2 d"],
    ]:
        """Compute soft alignment between two hypergraphs.

        Args:
            hg1: Source hypergraph (``n1`` nodes, dim ``d``).
            hg2: Target hypergraph (``n2`` nodes).

        Returns:
            Tuple ``(P, aligned_features)`` where ``P`` is the
            transport plan ``(n1, n2)`` and
            ``aligned_features`` are ``hg1`` features transported
            to ``hg2`` node space ``(n2, d)``.
        """
        X = hg1.node_features  # (n1, d)
        Y = hg2.node_features  # (n2, d)
        n1 = X.shape[0]
        n2 = Y.shape[0]

        a = jnp.ones(n1) / n1
        b = jnp.ones(n2) / n2

        C = feature_cost_matrix(
            X, Y, metric="sqeuclidean"
        )

        P = _sinkhorn_core(
            C, a, b,
            epsilon=self.epsilon,
            max_iters=self.max_iters,
        )

        col_mass = jnp.maximum(
            jnp.sum(P, axis=0), 1e-30
        )
        aligned = (P.T @ X) / col_mass[:, None]

        return P, aligned


# ---------------------------------------------------------------------------
# 9. OTConv -- optimal transport convolution layer
# ---------------------------------------------------------------------------


class OTConv(AbstractHypergraphConv):
    """Optimal transport convolution layer for hypergraphs.

    Replaces standard mean/sum aggregation in two-stage message
    passing with Sinkhorn-based optimal transport.

    Attributes:
        linear_in: Input projection.
        linear_out: Output projection.
        log_epsilon: Log of Sinkhorn regularisation (learnable).
    """

    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    log_epsilon: Float[Array, ""]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialise OTConv.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            key: PRNG key for weight initialisation.
        """
        k1, k2 = jax.random.split(key)
        self.linear_in = eqx.nn.Linear(
            in_dim, out_dim, key=k1
        )
        self.linear_out = eqx.nn.Linear(
            out_dim, out_dim, key=k2
        )
        self.log_epsilon = jnp.array(-2.3)

    def __call__(
        self,
        hg: Hypergraph,
    ) -> Float[Array, "n out_dim"]:
        """Apply OT convolution.

        Args:
            hg: Input hypergraph.

        Returns:
            Updated node features ``(num_nodes, out_dim)``.
        """
        H = hg._masked_incidence()
        eps = jnp.exp(self.log_epsilon)

        x = jax.vmap(self.linear_in)(hg.node_features)

        edge_features = ot_hyperedge_aggregation(
            x, H, epsilon=eps
        )

        def aggregate_to_node(node_row):
            membership = node_row
            degree = jnp.sum(membership)
            a_e = membership / jnp.maximum(
                degree, 1e-30
            )
            b_single = jnp.ones(1)

            candidate = jnp.sum(
                edge_features * membership[:, None],
                axis=0,
                keepdims=True,
            )
            diff = edge_features - candidate
            cost = jnp.sum(
                diff**2, axis=-1, keepdims=True
            )

            T, _ = _sinkhorn_with_cost(
                a_e, b_single, cost,
                epsilon=eps, num_iters=10,
            )
            agg = T[:, 0] @ edge_features
            return agg

        node_out = jax.vmap(
            aggregate_to_node, in_axes=0
        )(H)

        out = jax.vmap(self.linear_out)(node_out)

        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out
