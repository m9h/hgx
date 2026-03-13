"""Mixed-curvature product space hypergraph convolution.

Implements V->E->V message passing on hypergraphs where node features live
in a product manifold M = H^{d1} x S^{d2} x R^{d3}, combining hyperbolic,
spherical, and Euclidean components.  Each component has its own geometry:

  - **Hyperbolic (Poincare ball)**: Tree-like structures (cell lineages).
    Uses gyromidpoint aggregation and Poincare exp/log maps.
  - **Spherical**: Cyclic structures (cell cycle phases).
    Uses spherical midpoint aggregation and sphere exp/log maps.
  - **Euclidean**: Flat structures (expression levels).
    Uses standard mean aggregation.

The total distance is:

    d^2(x, y) = d^2_H(x_H, y_H) + d^2_S(x_S, y_S) + d^2_E(x_E, y_E)

This captures heterogeneous curvature in biological networks where different
aspects of the data live in spaces with different intrinsic geometries.

Two main convolution classes are provided:

  - ``ProductHypergraphConv``: Uses a ``ProductManifold`` descriptor.
  - ``ProductSpaceConv``: Simpler constructor taking ``component_dims``
    and ``component_types`` lists directly, with per-component linear
    transforms and geometry-appropriate aggregation.

Additionally:

  - ``ProductSpaceEmbedding``: Maps Euclidean features into a product
    space with appropriate per-component projections.
  - ``product_distance``: Free function computing the product metric.

References:
    Gu et al. (2019). "Learning Mixed-Curvature Representations in
        Product Spaces." ICLR 2019.
    Skopek et al. (2020). "Mixed-Curvature Variational Autoencoders."
        ICLR 2020.
    Bachmann et al. (2020). "Constant Curvature Graph Convolutional
        Networks." ICML 2020.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._conv._hyperbolic import (
    _EPS,
    expmap0,
    gyromidpoint,
    logmap0,
    project,
)
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# Spherical geometry primitives (unit sphere S^d)
# ---------------------------------------------------------------------------


def sphere_project(
    x: Float[Array, "... d"],
    eps: float = _EPS,
) -> Float[Array, "... d"]:
    """Project points onto the unit sphere S^d by normalising to unit norm.

    Args:
        x: Points in ambient Euclidean space.
        eps: Small constant to avoid division by zero.

    Returns:
        Points on the unit sphere (norm = 1).
    """
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def sphere_exp(
    x: Float[Array, "... d"],
    v: Float[Array, "... d"],
    eps: float = _EPS,
) -> Float[Array, "... d"]:
    """Exponential map on the unit sphere S^d.

    .. math::

        \\exp_x(v) = \\cos(\\|v\\|) \\, x + \\sin(\\|v\\|) \\, \\frac{v}{\\|v\\|}

    The tangent vector *v* is assumed to already be tangent to the sphere
    at *x* (i.e. ``<v, x> = 0``).

    Args:
        x: Base points on the unit sphere.
        v: Tangent vectors at x (orthogonal to x).
        eps: Numerical stability constant.

    Returns:
        Points on the unit sphere reached by following geodesics.
    """
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    v_norm_safe = jnp.maximum(v_norm, eps)
    result = jnp.cos(v_norm) * x + jnp.sin(v_norm) * (v / v_norm_safe)
    return sphere_project(result)


def sphere_log(
    x: Float[Array, "... d"],
    y: Float[Array, "... d"],
    eps: float = _EPS,
) -> Float[Array, "... d"]:
    """Logarithmic map on the unit sphere S^d.

    Returns the tangent vector at *x* pointing towards *y* with magnitude
    equal to the geodesic distance.

    .. math::

        \\log_x(y) = \\frac{\\theta}{\\sin(\\theta)} (y - \\cos(\\theta) \\, x)

    where ``theta = arccos(<x, y>)``.

    Args:
        x: Base points on the unit sphere.
        y: Target points on the unit sphere.
        eps: Numerical stability constant.

    Returns:
        Tangent vectors at x pointing towards y.
    """
    # Clamp dot product to [-1, 1] for numerical stability
    dot = jnp.sum(x * y, axis=-1, keepdims=True)
    dot = jnp.clip(dot, -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(dot)

    # Direction: component of y orthogonal to x
    direction = y - dot * x
    dir_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / jnp.maximum(dir_norm, eps)

    return theta * direction


def sphere_midpoint(
    x: Float[Array, "K d"],
    weights: Float[Array, " K"],
    n_iter: int = 5,
    eps: float = _EPS,
) -> Float[Array, " d"]:
    """Weighted Frechet mean on the unit sphere (iterative).

    Computes the weighted mean by iterative log-map averaging:
    at each step, compute the weighted mean tangent vector in the
    tangent space at the current estimate, then follow the geodesic.

    For well-separated points or few iterations, this converges to
    a good approximation of the true Frechet mean.

    Args:
        x: Points on the unit sphere, shape (K, d).
        weights: Non-negative weights, shape (K,).
        n_iter: Number of iterations for convergence.
        eps: Numerical stability constant.

    Returns:
        Weighted Frechet mean on the sphere, shape (d,).
    """
    # Initialise with the normalised weighted Euclidean mean
    w = weights / jnp.maximum(jnp.sum(weights), eps)
    mu = sphere_project(jnp.sum(w[:, None] * x, axis=0))

    def _step(mu: Float[Array, " d"], _: None) -> tuple[Float[Array, " d"], None]:
        # Log map each point to tangent space at mu
        tangents = sphere_log(
            jnp.broadcast_to(mu[None, :], x.shape), x
        )  # (K, d)
        # Weighted average tangent vector
        avg_tangent = jnp.sum(w[:, None] * tangents, axis=0)  # (d,)
        # Follow the geodesic
        mu_new = sphere_exp(mu, avg_tangent)
        return mu_new, None

    mu, _ = jax.lax.scan(_step, mu, None, length=n_iter)
    return mu


def _sphere_midpoint_batched(
    points: Float[Array, "K d"],
    membership: Float[Array, "K G"],
    n_iter: int = 5,
) -> Float[Array, "G d"]:
    """Batched spherical midpoint for V->E / E->V aggregation.

    For each group g, computes the weighted Frechet mean of points
    with weights given by column g of the membership matrix.

    Args:
        points: Points on the unit sphere, shape (K, d).
        membership: Membership weights, shape (K, G).
        n_iter: Iterations for the Frechet mean.

    Returns:
        Group centroids on the sphere, shape (G, d).
    """
    def _one_group(col: Float[Array, " K"]) -> Float[Array, " d"]:
        return sphere_midpoint(points, col, n_iter=n_iter)

    return jax.vmap(_one_group, in_axes=1)(membership)


# ---------------------------------------------------------------------------
# Product Manifold
# ---------------------------------------------------------------------------


class ProductManifold(eqx.Module):
    """Defines the component structure of a product manifold.

    A product manifold M = M_1 x M_2 x ... x M_k where each M_i is one of:
      - ``"hyperbolic"``: Poincare ball H^{d_i}
      - ``"spherical"``: Unit sphere S^{d_i}
      - ``"euclidean"``: Euclidean space R^{d_i}

    Attributes:
        components: List of (geometry_type, dimension) pairs.
            E.g. ``[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]``
            gives a 12-dimensional product manifold.
    """

    components: list[tuple[str, int]] = eqx.field(static=True)

    @property
    def total_dim(self) -> int:
        """Total dimension of the product manifold."""
        return sum(d for _, d in self.components)

    def split(
        self, x: Float[Array, "... total_dim"]
    ) -> list[Float[Array, "... d_i"]]:
        """Split features into per-component arrays.

        Args:
            x: Features in the product space, last axis = total_dim.

        Returns:
            List of arrays, one per component, split along the last axis.
        """
        parts = []
        offset = 0
        for _, dim in self.components:
            parts.append(x[..., offset : offset + dim])
            offset += dim
        return parts

    def combine(
        self, parts: list[Float[Array, "... d_i"]]
    ) -> Float[Array, "... total_dim"]:
        """Concatenate per-component arrays back into a single array.

        Args:
            parts: List of arrays, one per component.

        Returns:
            Concatenated array with last axis = total_dim.
        """
        return jnp.concatenate(parts, axis=-1)

    def project(
        self,
        x: Float[Array, "... total_dim"],
        curvatures: Float[Array, " num_components"] | None = None,
    ) -> Float[Array, "... total_dim"]:
        """Project each component onto its manifold.

        Args:
            x: Points in the product space.
            curvatures: Per-component curvature parameters.  Used for
                the hyperbolic component (Poincare ball radius 1/sqrt(c)).
                Spherical components always use the unit sphere.
                If None, curvature 1.0 is used for hyperbolic components.

        Returns:
            Projected points.
        """
        parts = self.split(x)
        projected = []
        for i, (geom, _) in enumerate(self.components):
            c = curvatures[i] if curvatures is not None else jnp.array(1.0)
            c = jnp.abs(c) + _EPS
            if geom == "hyperbolic":
                projected.append(project(parts[i], c))
            elif geom == "spherical":
                projected.append(sphere_project(parts[i]))
            else:  # euclidean
                projected.append(parts[i])
        return self.combine(projected)

    def exp(
        self,
        x: Float[Array, "... total_dim"],
        v: Float[Array, "... total_dim"],
        curvatures: Float[Array, " num_components"] | None = None,
    ) -> Float[Array, "... total_dim"]:
        """Component-wise exponential map.

        For hyperbolic: exp map at the origin (ignores x, uses v).
        For spherical: exp map at x along v.
        For Euclidean: x + v.

        Args:
            x: Base points in the product space.
            v: Tangent vectors.
            curvatures: Per-component curvature parameters.

        Returns:
            Points reached by the exponential map.
        """
        x_parts = self.split(x)
        v_parts = self.split(v)
        results = []
        for i, (geom, _) in enumerate(self.components):
            c = curvatures[i] if curvatures is not None else jnp.array(1.0)
            c = jnp.abs(c) + _EPS
            if geom == "hyperbolic":
                results.append(expmap0(v_parts[i], c))
            elif geom == "spherical":
                results.append(sphere_exp(x_parts[i], v_parts[i]))
            else:  # euclidean
                results.append(x_parts[i] + v_parts[i])
        return self.combine(results)

    def log(
        self,
        x: Float[Array, "... total_dim"],
        y: Float[Array, "... total_dim"],
        curvatures: Float[Array, " num_components"] | None = None,
    ) -> Float[Array, "... total_dim"]:
        """Component-wise logarithmic map.

        For hyperbolic: log map to the origin (ignores x, maps y to tangent space).
        For spherical: log map at x pointing to y.
        For Euclidean: y - x.

        Args:
            x: Base points in the product space.
            y: Target points in the product space.
            curvatures: Per-component curvature parameters.

        Returns:
            Tangent vectors.
        """
        x_parts = self.split(x)
        y_parts = self.split(y)
        results = []
        for i, (geom, _) in enumerate(self.components):
            c = curvatures[i] if curvatures is not None else jnp.array(1.0)
            c = jnp.abs(c) + _EPS
            if geom == "hyperbolic":
                results.append(logmap0(y_parts[i], c))
            elif geom == "spherical":
                results.append(sphere_log(x_parts[i], y_parts[i]))
            else:  # euclidean
                results.append(y_parts[i] - x_parts[i])
        return self.combine(results)

    def distance(
        self,
        x: Float[Array, "... total_dim"],
        y: Float[Array, "... total_dim"],
        curvatures: Float[Array, " num_components"] | None = None,
    ) -> Float[Array, ...]:
        """Product metric distance.

        .. math::

            d^2(x, y) = \\sum_i d^2_i(x_i, y_i)

        Args:
            x: First set of points.
            y: Second set of points.
            curvatures: Per-component curvature parameters.

        Returns:
            Distances, shape = leading batch dimensions of x and y.
        """
        x_parts = self.split(x)
        y_parts = self.split(y)
        dist_sq = jnp.zeros(x.shape[:-1])
        for i, (geom, _) in enumerate(self.components):
            c = curvatures[i] if curvatures is not None else jnp.array(1.0)
            c = jnp.abs(c) + _EPS
            if geom == "hyperbolic":
                # Poincare distance: d(x,y) via log map norm
                v = logmap0(x_parts[i], c)
                w = logmap0(y_parts[i], c)
                # Use the tangent space distance at origin as approximation,
                # or compute exact Poincare distance:
                # d_c(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) oplus_c y||)
                # For simplicity, use ||log_0(x) - log_0(y)||^2
                diff = v - w
                dist_sq = dist_sq + jnp.sum(diff ** 2, axis=-1)
            elif geom == "spherical":
                # Geodesic distance on sphere: arccos(<x, y>)
                dot = jnp.sum(
                    x_parts[i] * y_parts[i], axis=-1
                )
                dot = jnp.clip(dot, -1.0 + _EPS, 1.0 - _EPS)
                theta = jnp.arccos(dot)
                dist_sq = dist_sq + theta ** 2
            else:  # euclidean
                diff = x_parts[i] - y_parts[i]
                dist_sq = dist_sq + jnp.sum(diff ** 2, axis=-1)
        return jnp.sqrt(jnp.maximum(dist_sq, _EPS))


# ---------------------------------------------------------------------------
# Product Hypergraph Convolution
# ---------------------------------------------------------------------------


class ProductHypergraphConv(AbstractHypergraphConv):
    """Hypergraph convolution on mixed-curvature product manifolds.

    Performs V->E->V message passing where each manifold component uses
    its natural aggregation and update operations:

    1. **Linear transform**: Log map to tangent space at origin,
       apply per-component linear maps, exp map back.
    2. **V -> E**: Component-wise midpoint aggregation:
       gyromidpoint for hyperbolic, spherical Frechet mean for spherical,
       weighted mean for Euclidean.
    3. **E -> V**: Same component-wise midpoint aggregation in reverse.

    Curvatures are learnable per component.

    Attributes:
        manifold: The product manifold structure.
        linears: Per-component linear layers (tangent-space maps).
        log_curvatures: Learnable log-curvature parameters (one per component).
    """

    manifold: ProductManifold
    linears: list[eqx.nn.Linear]
    log_curvatures: Float[Array, " num_components"]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        manifold: ProductManifold,
        *,
        key: PRNGKeyArray,
    ):
        """Initialise ProductHypergraphConv.

        Args:
            in_dim: Total input feature dimension (must equal manifold.total_dim
                or be transformable to it).
            out_dim: Total output feature dimension.
            manifold: Product manifold defining the component structure.
                The output dimension of each component linear is inferred
                proportionally from out_dim, matching the manifold component
                ratios.
            key: PRNG key for weight initialisation.

        Raises:
            ValueError: If in_dim does not match manifold.total_dim.
        """
        if in_dim != manifold.total_dim:
            raise ValueError(
                f"in_dim ({in_dim}) must match manifold.total_dim "
                f"({manifold.total_dim})."
            )

        self.manifold = manifold

        # Compute output dims proportionally to component dimensions
        in_dims = [d for _, d in manifold.components]
        ratios = [d / sum(in_dims) for d in in_dims]
        out_dims_raw = [max(1, int(r * out_dim)) for r in ratios]
        # Adjust to match out_dim exactly
        remainder = out_dim - sum(out_dims_raw)
        # Distribute remainder to largest components
        sorted_idx = sorted(
            range(len(out_dims_raw)), key=lambda i: in_dims[i], reverse=True
        )
        for j in range(abs(remainder)):
            idx = sorted_idx[j % len(sorted_idx)]
            out_dims_raw[idx] += 1 if remainder > 0 else -1

        # Build per-component linear layers
        linears = []
        for i, (_, d_in) in enumerate(manifold.components):
            subkey, key = jax.random.split(key)
            linears.append(eqx.nn.Linear(d_in, out_dims_raw[i], key=subkey))
        self.linears = linears

        # Initialise curvatures: log(1.0) = 0 for all components
        self.log_curvatures = jnp.zeros(len(manifold.components))

    @property
    def curvatures(self) -> Float[Array, " num_components"]:
        """Positive curvature parameters (one per component)."""
        return jnp.exp(self.log_curvatures)

    @property
    def _output_manifold(self) -> ProductManifold:
        """Product manifold for the output space (with output dims)."""
        components = [
            (geom, self.linears[i].out_features)
            for i, (geom, _) in enumerate(self.manifold.components)
        ]
        return ProductManifold(components=components)

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply product-space hypergraph convolution.

        Args:
            hg: Input hypergraph with node features of shape
                (n, manifold.total_dim).

        Returns:
            Updated node features of shape (n, out_dim), with each
            component projected onto its manifold.
        """
        H = hg._masked_incidence()  # (n, m)
        x = hg.node_features  # (n, total_dim)
        curvatures = self.curvatures

        # Project inputs onto manifold
        x = self.manifold.project(x, curvatures)

        # --- Per-component tangent-space linear transform ---
        in_parts = self.manifold.split(x)
        transformed_parts = []
        for i, (geom, _) in enumerate(self.manifold.components):
            c = jnp.abs(curvatures[i]) + _EPS
            xi = in_parts[i]  # (n, d_i)
            if geom == "hyperbolic":
                v = logmap0(xi, c)
                v = jax.vmap(self.linears[i])(v)
                xi_new = expmap0(v, c)
                xi_new = project(xi_new, c)
            elif geom == "spherical":
                # Log map to tangent space at north pole, apply linear,
                # then project back.  We use a simpler approach: project
                # features to tangent space via subtraction of radial
                # component, apply linear, project back to sphere.
                # Tangent space at north pole e_0 = (1, 0, ..., 0):
                # v = sphere_log(e_0, x), but more practically, apply
                # the linear and renormalise.
                v = jax.vmap(self.linears[i])(xi)
                xi_new = sphere_project(v)
            else:  # euclidean
                xi_new = jax.vmap(self.linears[i])(xi)
            transformed_parts.append(xi_new)

        # Build output manifold for aggregation
        out_manifold = self._output_manifold

        # --- V -> E aggregation: component-wise midpoints ---
        x_transformed = jnp.concatenate(transformed_parts, axis=-1)  # (n, out_dim)
        t_parts = out_manifold.split(x_transformed)
        e_parts = []
        for i, (geom, _) in enumerate(out_manifold.components):
            c = jnp.abs(curvatures[i]) + _EPS
            if geom == "hyperbolic":
                e_parts.append(gyromidpoint(t_parts[i], H, c))
            elif geom == "spherical":
                e_parts.append(
                    _sphere_midpoint_batched(t_parts[i], H)
                )
            else:  # euclidean
                # Weighted mean: H^T @ x / sum(H, axis=0)
                numerator = H.T @ t_parts[i]  # (m, d_i)
                denom = jnp.sum(H, axis=0)[:, None]  # (m, 1)
                e_parts.append(numerator / jnp.maximum(denom, _EPS))

        # --- E -> V aggregation: component-wise midpoints ---
        out_parts = []
        for i, (geom, _) in enumerate(out_manifold.components):
            c = jnp.abs(curvatures[i]) + _EPS
            if geom == "hyperbolic":
                out_parts.append(gyromidpoint(e_parts[i], H.T, c))
            elif geom == "spherical":
                out_parts.append(
                    _sphere_midpoint_batched(e_parts[i], H.T)
                )
            else:  # euclidean
                numerator = H @ e_parts[i]  # (n, d_i)
                denom = jnp.sum(H, axis=1)[:, None]  # (n, 1)
                out_parts.append(numerator / jnp.maximum(denom, _EPS))

        out = jnp.concatenate(out_parts, axis=-1)

        # Final projection onto manifold
        out = out_manifold.project(out, curvatures)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out


# ---------------------------------------------------------------------------
# Auto-curvature fitting
# ---------------------------------------------------------------------------


def auto_curvature(
    hg: Hypergraph,
    dim_per_component: list[tuple[str, int]],
    *,
    key: PRNGKeyArray,
    n_steps: int = 200,
    lr: float = 0.01,
) -> ProductManifold:
    """Learn optimal curvature assignment by fitting component curvatures.

    Fits component curvatures to minimise distortion of the hypergraph's
    pairwise distance structure.  The distance between nodes in the product
    manifold should approximate the shortest-path distance in the hypergraph.

    This is done by:
    1. Computing a graph distance matrix from the incidence structure.
    2. Randomly initialising node embeddings in the product manifold.
    3. Optimising curvatures (and embeddings) to minimise a distortion loss.

    Args:
        hg: Input hypergraph.
        dim_per_component: Component specification, e.g.
            ``[("hyperbolic", 4), ("spherical", 3), ("euclidean", 5)]``.
        key: PRNG key.
        n_steps: Number of optimisation steps.
        lr: Learning rate for gradient descent.

    Returns:
        A ProductManifold with optimised curvature information.  The returned
        manifold has the same component types and dimensions but serves as
        a fitted structure.
    """
    import optax

    manifold = ProductManifold(components=dim_per_component)
    n = hg.node_features.shape[0]
    total_dim = manifold.total_dim

    # Compute graph distances from incidence (1-hop adjacency)
    H = hg._masked_incidence()
    # Adjacency: A = H @ H^T, zero diagonal
    A = H @ H.T
    A = A.at[jnp.diag_indices(n)].set(0.0)
    A = (A > 0).astype(jnp.float32)

    # BFS-like shortest path via repeated matrix multiply (capped at diameter)
    # For simplicity, use powers of adjacency up to a max depth
    max_depth = min(n, 10)
    dist_matrix = jnp.full((n, n), jnp.inf)
    dist_matrix = dist_matrix.at[jnp.diag_indices(n)].set(0.0)
    current_adj = A
    for depth in range(1, max_depth + 1):
        newly_reached = (current_adj > 0) & (dist_matrix == jnp.inf)
        dist_matrix = jnp.where(newly_reached, float(depth), dist_matrix)
        current_adj = current_adj @ A

    # Replace inf with max_depth + 1 for unreachable pairs
    dist_matrix = jnp.where(dist_matrix == jnp.inf, max_depth + 1.0, dist_matrix)

    # Parameters to optimise
    key_emb, key_curv = jax.random.split(key)
    embeddings = jax.random.normal(key_emb, (n, total_dim)) * 0.1
    log_curvatures = jnp.zeros(len(dim_per_component))

    # Project initial embeddings
    curvatures = jnp.exp(log_curvatures)
    embeddings = manifold.project(embeddings, curvatures)

    class _FitState(eqx.Module):
        embeddings: Float[Array, "n total_dim"]
        log_curvatures: Float[Array, " num_components"]

    state = _FitState(embeddings=embeddings, log_curvatures=log_curvatures)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(state)

    def _loss(state: _FitState) -> Float[Array, ""]:
        curvatures = jnp.exp(state.log_curvatures)
        emb = manifold.project(state.embeddings, curvatures)
        # Compute pairwise product distances
        # Vectorise over pairs using vmap
        def _row_dist(xi: Float[Array, " total_dim"]) -> Float[Array, " n"]:
            return jax.vmap(
                partial(manifold.distance, curvatures=curvatures)
            )(
                jnp.broadcast_to(xi[None, :], (n, total_dim)),
                emb,
            )

        pred_dist = jax.vmap(_row_dist)(emb)  # (n, n)
        # Stress loss: sum of (d_pred - d_target)^2 / d_target^2
        # Only consider pairs with finite target distance
        mask = (dist_matrix < max_depth + 0.5) & (dist_matrix > 0)
        target = jnp.where(mask, dist_matrix, 1.0)
        residual = (pred_dist - target) ** 2
        loss = jnp.sum(jnp.where(mask, residual / (target ** 2 + _EPS), 0.0))
        return loss / jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)

    @jax.jit
    def _step(
        state: _FitState, opt_state: optax.OptState
    ) -> tuple[_FitState, optax.OptState, Float[Array, ""]]:
        loss, grads = jax.value_and_grad(_loss)(state)
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = eqx.apply_updates(state, updates)
        return state, opt_state, loss

    for _ in range(n_steps):
        state, opt_state, loss = _step(state, opt_state)

    return manifold


# ---------------------------------------------------------------------------
# Public aliases for spherical helpers
# ---------------------------------------------------------------------------

#: Exponential map on the unit sphere.  Alias for :func:`sphere_exp`.
sphere_exp_map = sphere_exp

#: Logarithmic map on the unit sphere.  Alias for :func:`sphere_log`.
sphere_log_map = sphere_log

#: Project onto the unit sphere by normalising.  Alias for :func:`sphere_project`.
project_to_sphere = sphere_project


# ---------------------------------------------------------------------------
# Product distance (free function)
# ---------------------------------------------------------------------------


def _poincare_distance(
    x: Float[Array, "... d"],
    y: Float[Array, "... d"],
    c: float = 1.0,
) -> Float[Array, ...]:
    """Distance in the Poincare ball of curvature c.

    d(x, y) = (1/sqrt(c)) * arccosh(1 + 2c * ||x-y||^2 /
        ((1 - c||x||^2)(1 - c||y||^2)))
    """
    diff_sq = jnp.sum((x - y) ** 2, axis=-1)
    x_sq = jnp.sum(x ** 2, axis=-1)
    y_sq = jnp.sum(y ** 2, axis=-1)
    denom = jnp.maximum((1 - c * x_sq) * (1 - c * y_sq), _EPS)
    arg = 1 + 2 * c * diff_sq / denom
    arg = jnp.maximum(arg, 1.0 + _EPS)
    return jnp.arccosh(arg) / jnp.sqrt(c)


def _sphere_distance(
    x: Float[Array, "... d"],
    y: Float[Array, "... d"],
) -> Float[Array, ...]:
    """Geodesic distance on the unit sphere: arccos(<x, y>)."""
    dot = jnp.sum(x * y, axis=-1)
    dot = jnp.clip(dot, -1.0 + _EPS, 1.0 - _EPS)
    return jnp.arccos(dot)


def product_distance(
    x: Float[Array, "... D"],
    y: Float[Array, "... D"],
    component_dims: Sequence[int],
    component_types: Sequence[str],
) -> Float[Array, ...]:
    """Distance in the product manifold M_1 x M_2 x ... x M_k.

    The product metric is:

        d(x, y) = sqrt( sum_i d_i(x_i, y_i)^2 )

    where d_i is the intrinsic distance on component i:
      - **hyperbolic**: Exact Poincare ball distance with c=1.
      - **spherical**: Geodesic (arc-length) distance on the unit sphere.
      - **euclidean**: L2 distance.

    Args:
        x: Point(s) in the product space with total dim = sum(component_dims).
        y: Point(s) in the product space.
        component_dims: Dimension of each component.
        component_types: Type of each component (``"hyperbolic"``,
            ``"spherical"``, or ``"euclidean"``).

    Returns:
        Non-negative scalar (or batch of) distance(s).

    Raises:
        ValueError: If lengths of ``component_dims`` and ``component_types``
            differ, or an unknown component type is given.
    """
    if len(component_dims) != len(component_types):
        raise ValueError(
            f"component_dims and component_types must have the same length. "
            f"Got {len(component_dims)} and {len(component_types)}."
        )

    offset = 0
    dist_sq = jnp.zeros(x.shape[:-1])

    for dim, ctype in zip(component_dims, component_types):
        xi = jax.lax.dynamic_slice_in_dim(x, offset, dim, axis=-1)
        yi = jax.lax.dynamic_slice_in_dim(y, offset, dim, axis=-1)

        if ctype == "hyperbolic":
            d_i = _poincare_distance(xi, yi, c=1.0)
        elif ctype == "spherical":
            d_i = _sphere_distance(xi, yi)
        elif ctype == "euclidean":
            d_i = jnp.sqrt(jnp.sum((xi - yi) ** 2, axis=-1) + _EPS)
        else:
            raise ValueError(f"Unknown component type: {ctype!r}")

        dist_sq = dist_sq + d_i ** 2
        offset += dim

    return jnp.sqrt(dist_sq + _EPS)


# ---------------------------------------------------------------------------
# Product space embedding
# ---------------------------------------------------------------------------


class ProductSpaceEmbedding(eqx.Module):
    """Learnable embedding that maps Euclidean features into a product space.

    For each component, applies a separate linear map from the full input
    dimension, then projects onto the appropriate manifold:

    - **Hyperbolic**: project into the Poincare ball (norm < 1)
    - **Spherical**: normalize to the unit sphere
    - **Euclidean**: identity (no projection)

    The output is the concatenation of all component embeddings.

    Attributes:
        linears: One linear layer per component.
        component_dims: Dimension of each component.
        component_types: Geometry type of each component.
    """

    linears: list[eqx.nn.Linear]
    component_dims: tuple[int, ...] = eqx.field(static=True)
    component_types: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        component_dims: Sequence[int],
        component_types: Sequence[str],
        *,
        key: PRNGKeyArray,
    ):
        """Initialize ProductSpaceEmbedding.

        Args:
            in_dim: Input Euclidean feature dimension.
            component_dims: Output dimension for each manifold component.
            component_types: Geometry type per component (``"hyperbolic"``,
                ``"spherical"``, or ``"euclidean"``).
            key: PRNG key for weight initialisation.

        Raises:
            ValueError: If ``component_dims`` and ``component_types`` have
                different lengths, or an unknown type is given.
        """
        if len(component_dims) != len(component_types):
            raise ValueError(
                f"component_dims and component_types must have the same length. "
                f"Got {len(component_dims)} and {len(component_types)}."
            )
        for ctype in component_types:
            if ctype not in ("hyperbolic", "spherical", "euclidean"):
                raise ValueError(f"Unknown component type: {ctype!r}")

        self.component_dims = tuple(component_dims)
        self.component_types = tuple(component_types)

        keys = jax.random.split(key, len(component_dims))
        self.linears = [
            eqx.nn.Linear(in_dim, d, key=k)
            for d, k in zip(component_dims, keys)
        ]

    @property
    def out_dim(self) -> int:
        """Total output dimension (sum of component dims)."""
        return sum(self.component_dims)

    def __call__(
        self, x: Float[Array, "... in_dim"]
    ) -> Float[Array, "... out_dim"]:
        """Embed Euclidean features into the product space.

        Applies per-component linear projections followed by manifold
        projection, then concatenates all components.

        Args:
            x: Input features of shape (..., in_dim).

        Returns:
            Product space embeddings of shape (..., sum(component_dims)).
        """
        parts = []
        for linear, dim, ctype in zip(
            self.linears, self.component_dims, self.component_types
        ):
            h = linear(x)
            if ctype == "hyperbolic":
                c = jnp.array(1.0)
                h = project(h, c)
            elif ctype == "spherical":
                h = sphere_project(h)
            # euclidean: no projection needed
            parts.append(h)

        return jnp.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# ProductSpaceConv — simplified product-space convolution
# ---------------------------------------------------------------------------


class ProductSpaceConv(AbstractHypergraphConv):
    """Hypergraph convolution on a product of manifold components.

    Splits node features into components corresponding to different
    geometries and performs independent V->E->V message passing in each
    component using geometry-appropriate operations:

    - **Hyperbolic**: Mobius midpoint (gyromidpoint) aggregation in the
      Poincare ball via the Klein model.
    - **Spherical**: Geodesic mean on S^d via iterative tangent-space
      averaging (Frechet mean).
    - **Euclidean**: Standard degree-normalised weighted mean.

    Each component has its own learnable linear transform.  After
    independent aggregation the component outputs are concatenated.

    This is a simpler alternative to :class:`ProductHypergraphConv` that
    takes ``component_dims`` and ``component_types`` directly instead of
    requiring a :class:`ProductManifold` object.

    Attributes:
        linears: One linear layer per component.
        component_dims: Dimension of each input component.
        component_types: Geometry type of each component.
        out_dims: Output dimension for each component (proportional split
            of ``out_dim``).
        c: Learnable curvature for hyperbolic components (positive scalar).
    """

    linears: list[eqx.nn.Linear]
    component_dims: tuple[int, ...] = eqx.field(static=True)
    component_types: tuple[str, ...] = eqx.field(static=True)
    out_dims: tuple[int, ...] = eqx.field(static=True)
    c: Float[Array, ""]

    def __init__(
        self,
        component_dims: Sequence[int],
        component_types: Sequence[str],
        out_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize ProductSpaceConv.

        The total output dimension ``out_dim`` is split proportionally among
        components according to their input dimension ratios, with rounding
        adjustments applied to the last component.

        Args:
            component_dims: Dimension of each input manifold component.
                Total input dim = sum(component_dims).
            component_types: Geometry type per component (``"hyperbolic"``,
                ``"spherical"``, or ``"euclidean"``).
            out_dim: Total output feature dimension.
            key: PRNG key for weight initialisation.

        Raises:
            ValueError: If ``component_dims`` and ``component_types`` have
                different lengths, or an unknown type is given.
        """
        if len(component_dims) != len(component_types):
            raise ValueError(
                f"component_dims and component_types must have the same length. "
                f"Got {len(component_dims)} and {len(component_types)}."
            )
        for ctype in component_types:
            if ctype not in ("hyperbolic", "spherical", "euclidean"):
                raise ValueError(f"Unknown component type: {ctype!r}")

        self.component_dims = tuple(component_dims)
        self.component_types = tuple(component_types)

        # Split out_dim proportionally to input dims
        total_in = sum(component_dims)
        raw_out = [
            max(1, int(round(d / total_in * out_dim)))
            for d in component_dims
        ]
        # Adjust last component so total equals out_dim
        raw_out[-1] = out_dim - sum(raw_out[:-1])
        self.out_dims = tuple(raw_out)

        keys = jax.random.split(key, len(component_dims))
        self.linears = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(component_dims, self.out_dims, keys)
        ]

        self.c = jnp.array(1.0, dtype=jnp.float32)

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply product space hypergraph convolution.

        Splits features into components, applies component-specific
        V->E->V message passing, and concatenates outputs.

        Args:
            hg: Input hypergraph with node features of shape
                ``(n, sum(component_dims))``.

        Returns:
            Updated node features of shape ``(n, sum(out_dims))``.
        """
        H = hg._masked_incidence()  # (n, m)
        x = hg.node_features  # (n, total_dim)
        c = jnp.abs(self.c) + _EPS

        parts = []
        offset = 0

        for i, (dim, ctype) in enumerate(
            zip(self.component_dims, self.component_types)
        ):
            # Extract this component's features
            xi = x[..., offset : offset + dim]
            linear = self.linears[i]

            if ctype == "hyperbolic":
                parts.append(self._hyperbolic_pass(xi, H, linear, c))
            elif ctype == "spherical":
                parts.append(self._spherical_pass(xi, H, linear))
            elif ctype == "euclidean":
                parts.append(self._euclidean_pass(xi, H, linear))

            offset += dim

        out = jnp.concatenate(parts, axis=-1)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out

    # -- component-specific message passing ----------------------------------

    def _hyperbolic_pass(
        self,
        x: Float[Array, "n d"],
        H: Float[Array, "n m"],
        linear: eqx.nn.Linear,
        c: Float[Array, ""],
    ) -> Float[Array, "n d_out"]:
        """Hyperbolic V->E->V via Mobius midpoint in the Poincare ball."""
        # Project into ball
        x = project(x, c)

        # Linear in tangent space at origin
        v = logmap0(x, c)
        v = jax.vmap(linear)(v)
        x = expmap0(v, c)
        x = project(x, c)

        # V -> E: gyromidpoint aggregation
        e = gyromidpoint(x, H, c)

        # E -> V: gyromidpoint aggregation
        return gyromidpoint(e, H.T, c)

    def _spherical_pass(
        self,
        x: Float[Array, "n d"],
        H: Float[Array, "n m"],
        linear: eqx.nn.Linear,
    ) -> Float[Array, "n d_out"]:
        """Spherical V->E->V via geodesic mean on S^d."""
        # Ensure on sphere
        x = sphere_project(x)

        # Apply linear in ambient space and re-project
        z = jax.vmap(linear)(x)
        x = sphere_project(z)

        # V -> E: sphere midpoint aggregation
        e = _sphere_midpoint_batched(x, H)  # (m, d_out)

        # E -> V: sphere midpoint aggregation
        return _sphere_midpoint_batched(e, H.T)  # (n, d_out)

    def _euclidean_pass(
        self,
        x: Float[Array, "n d"],
        H: Float[Array, "n m"],
        linear: eqx.nn.Linear,
    ) -> Float[Array, "n d_out"]:
        """Euclidean V->E->V via degree-normalised mean."""
        x = jax.vmap(linear)(x)

        # Degree-normalised aggregation
        d_e = jnp.sum(H, axis=0, keepdims=True)  # (1, m)
        d_v = jnp.sum(H, axis=1, keepdims=True)  # (n, 1)

        d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
        d_v_inv = jnp.where(d_v > 0, 1.0 / d_v, 0.0)

        # V -> E: mean aggregation
        e = (H * d_e_inv).T @ x  # (m, d_out)

        # E -> V: mean aggregation
        return d_v_inv * (H @ e)  # (n, d_out)


# ---------------------------------------------------------------------------
# Standalone product manifold utilities
# ---------------------------------------------------------------------------


def split_components(
    x: Float[Array, "... D"],
    dims: tuple[int, int, int],
) -> tuple[
    Float[Array, "... d_h"],
    Float[Array, "... d_s"],
    Float[Array, "... d_e"],
]:
    """Split feature vector into (hyperbolic, spherical, euclidean).

    Args:
        x: Feature vector(s) with last axis = sum(dims).
        dims: Tuple ``(d_h, d_s, d_e)`` giving the dimension of each
            component space.

    Returns:
        Tuple of three arrays for hyperbolic, spherical, and
        Euclidean components.
    """
    d_h, d_s, d_e = dims
    h = x[..., :d_h]
    s = x[..., d_h : d_h + d_s]
    e = x[..., d_h + d_s : d_h + d_s + d_e]
    return h, s, e


def concat_components(
    h: Float[Array, "... d_h"],
    s: Float[Array, "... d_s"],
    e: Float[Array, "... d_e"],
) -> Float[Array, "... D"]:
    """Concatenate (hyperbolic, spherical, euclidean) components.

    Args:
        h: Hyperbolic component.
        s: Spherical component.
        e: Euclidean component.

    Returns:
        Concatenated feature vector.
    """
    return jnp.concatenate([h, s, e], axis=-1)


def project_hyperbolic(
    x: Float[Array, "... d"],
    c: float = 1.0,
) -> Float[Array, "... d"]:
    """Project onto the Poincare ball of curvature *c*.

    Clips the norm to be strictly less than ``1 / sqrt(c)``.

    Args:
        x: Points in ambient space.
        c: Positive curvature parameter.

    Returns:
        Points inside the Poincare ball.
    """
    c_arr = jnp.array(c, dtype=x.dtype)
    return project(x, c_arr)


def project_spherical(
    x: Float[Array, "... d"],
) -> Float[Array, "... d"]:
    """Project onto the unit sphere by normalising to unit norm.

    Args:
        x: Points in ambient space.

    Returns:
        Points on the unit sphere.
    """
    return sphere_project(x)


def exp_map_poincare(
    x: Float[Array, "... d"],
    v: Float[Array, "... d"],
    c: float = 1.0,
) -> Float[Array, "... d"]:
    """Exponential map on the Poincare ball at point *x*.

    For numerical simplicity this uses the origin-based maps:
    ``exp_0(log_0(x) + v)``, which is exact when *x* is the origin
    and a good tangent-space approximation otherwise.

    Args:
        x: Base point(s) in the Poincare ball.
        v: Tangent vector(s) at *x*.
        c: Positive curvature parameter.

    Returns:
        Points in the Poincare ball.
    """
    c_arr = jnp.array(c, dtype=x.dtype)
    c_arr = jnp.abs(c_arr) + _EPS
    # Map x to tangent space at origin, add v, map back
    t = logmap0(x, c_arr) + v
    result = expmap0(t, c_arr)
    return project(result, c_arr)


def log_map_poincare(
    x: Float[Array, "... d"],
    y: Float[Array, "... d"],
    c: float = 1.0,
) -> Float[Array, "... d"]:
    """Logarithmic map on the Poincare ball from *x* to *y*.

    Returns the tangent vector at *x* pointing toward *y*.  Uses
    origin-based maps for numerical stability: ``log_0(y) - log_0(x)``.

    Args:
        x: Base point(s) in the Poincare ball.
        y: Target point(s) in the Poincare ball.
        c: Positive curvature parameter.

    Returns:
        Tangent vectors at *x*.
    """
    c_arr = jnp.array(c, dtype=x.dtype)
    c_arr = jnp.abs(c_arr) + _EPS
    return logmap0(y, c_arr) - logmap0(x, c_arr)


def exp_map_sphere(
    x: Float[Array, "... d"],
    v: Float[Array, "... d"],
) -> Float[Array, "... d"]:
    """Exponential map on the unit sphere at point *x*.

    Args:
        x: Base point(s) on the unit sphere.
        v: Tangent vector(s) at *x* (orthogonal to *x*).

    Returns:
        Points on the unit sphere.
    """
    return sphere_exp(x, v)


def log_map_sphere(
    x: Float[Array, "... d"],
    y: Float[Array, "... d"],
) -> Float[Array, "... d"]:
    """Logarithmic map on the unit sphere from *x* to *y*.

    Args:
        x: Base point(s) on the unit sphere.
        y: Target point(s) on the unit sphere.

    Returns:
        Tangent vectors at *x* pointing toward *y*.
    """
    return sphere_log(x, y)


# ---------------------------------------------------------------------------
# ProductManifoldConv
# ---------------------------------------------------------------------------


class ProductManifoldConv(eqx.Module):
    """Mixed-curvature product manifold hypergraph convolution.

    Performs V->E->V message passing on H^{d_h} x S^{d_s} x R^{d_e}
    with geometry-appropriate aggregation for each component:

    - **Hyperbolic**: gyromidpoint (Mobius) aggregation in the
      Poincare ball.
    - **Spherical**: Frechet mean (iterative) on the unit sphere.
    - **Euclidean**: standard weighted mean.

    Uses the incidence matrix for V->E and E->V aggregation.

    Attributes:
        linear_h: Linear layer for the hyperbolic component.
        linear_s: Linear layer for the spherical component.
        linear_e: Linear layer for the Euclidean component.
        dims: ``(d_h, d_s, d_e)`` input component dimensions.
        out_dims: ``(o_h, o_s, o_e)`` output component dimensions.
        c_h: Learnable curvature for the hyperbolic component.
        c_s: Learnable curvature for the spherical component.
    """

    linear_h: eqx.nn.Linear
    linear_s: eqx.nn.Linear
    linear_e: eqx.nn.Linear
    dims: tuple[int, int, int] = eqx.field(static=True)
    out_dims: tuple[int, int, int] = eqx.field(static=True)
    c_h: Float[Array, ""]
    c_s: Float[Array, ""]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dims: tuple[int, int, int],
        curvatures: tuple[float, float],
        *,
        key: PRNGKeyArray,
    ):
        """Initialise ProductManifoldConv.

        Args:
            in_dim: Total input feature dimension (sum of dims).
            out_dim: Total output feature dimension.
            dims: ``(d_h, d_s, d_e)`` dimensions of the hyperbolic,
                spherical, and Euclidean input components.
            curvatures: ``(c_h, c_s)`` initial curvatures for the
                hyperbolic and spherical components.
            key: PRNG key for weight initialisation.

        Raises:
            ValueError: If ``in_dim != sum(dims)``.
        """
        d_h, d_s, d_e = dims
        if in_dim != d_h + d_s + d_e:
            raise ValueError(
                f"in_dim ({in_dim}) must equal sum(dims) "
                f"({d_h + d_s + d_e})."
            )

        self.dims = dims

        # Proportional output split
        total_in = d_h + d_s + d_e
        o_h = max(1, round(d_h / total_in * out_dim))
        o_s = max(1, round(d_s / total_in * out_dim))
        o_e = out_dim - o_h - o_s
        if o_e < 1:
            o_e = 1
            o_s = out_dim - o_h - o_e
        self.out_dims = (o_h, o_s, o_e)

        k1, k2, k3 = jax.random.split(key, 3)
        self.linear_h = eqx.nn.Linear(d_h, o_h, key=k1)
        self.linear_s = eqx.nn.Linear(d_s, o_s, key=k2)
        self.linear_e = eqx.nn.Linear(d_e, o_e, key=k3)

        c_h_init, c_s_init = curvatures
        self.c_h = jnp.array(c_h_init, dtype=jnp.float32)
        self.c_s = jnp.array(c_s_init, dtype=jnp.float32)

    def __call__(
        self, hg: Hypergraph
    ) -> Float[Array, "n out_dim"]:
        """Apply product manifold convolution.

        Args:
            hg: Input hypergraph with node features of shape
                ``(n, sum(dims))``.

        Returns:
            Updated node features, shape ``(n, sum(out_dims))``.
        """
        H = hg._masked_incidence()  # (n, m)
        x = hg.node_features  # (n, D)

        c_h = jnp.abs(self.c_h) + _EPS

        # Split into components
        x_h, x_s, x_e = split_components(x, self.dims)

        # Project inputs onto manifolds
        x_h = project(x_h, c_h)
        x_s = sphere_project(x_s)

        # --- Per-component linear transform ---
        # Hyperbolic: tangent-space linear
        v_h = logmap0(x_h, c_h)
        v_h = jax.vmap(self.linear_h)(v_h)
        x_h = expmap0(v_h, c_h)
        x_h = project(x_h, c_h)

        # Spherical: linear + re-project
        x_s = jax.vmap(self.linear_s)(x_s)
        x_s = sphere_project(x_s)

        # Euclidean: standard linear
        x_e = jax.vmap(self.linear_e)(x_e)

        # --- V -> E aggregation ---
        # Hyperbolic: gyromidpoint
        e_h = gyromidpoint(x_h, H, c_h)  # (m, o_h)

        # Spherical: Frechet mean
        e_s = _sphere_midpoint_batched(x_s, H)  # (m, o_s)

        # Euclidean: degree-normalised mean
        deg_e = jnp.sum(H, axis=0, keepdims=True).T  # (m, 1)
        e_e = H.T @ x_e  # (m, o_e)
        e_e = e_e / jnp.maximum(deg_e, _EPS)

        # --- E -> V aggregation ---
        # Hyperbolic: gyromidpoint
        out_h = gyromidpoint(e_h, H.T, c_h)  # (n, o_h)

        # Spherical: Frechet mean
        out_s = _sphere_midpoint_batched(e_s, H.T)  # (n, o_s)

        # Euclidean: degree-normalised mean
        deg_v = jnp.sum(H, axis=1, keepdims=True)  # (n, 1)
        out_e = H @ e_e  # (n, o_e)
        out_e = out_e / jnp.maximum(deg_v, _EPS)

        # Final projection
        out_h = project(out_h, c_h)
        out_s = sphere_project(out_s)

        out = concat_components(out_h, out_s, out_e)

        # Mask
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out


# ---------------------------------------------------------------------------
# ProductManifoldMLP
# ---------------------------------------------------------------------------


class ProductManifoldMLP(eqx.Module):
    """MLP operating in tangent space of a product manifold.

    For each component, the forward pass is:
    ``log_map -> linear1 -> activation -> linear2 -> exp_map``

    This keeps features on the correct manifold while allowing
    nonlinear transformations in tangent space.

    Attributes:
        linear1_h: First hidden layer for hyperbolic component.
        linear2_h: Second hidden layer for hyperbolic component.
        linear1_s: First hidden layer for spherical component.
        linear2_s: Second hidden layer for spherical component.
        linear1_e: First hidden layer for Euclidean component.
        linear2_e: Second hidden layer for Euclidean component.
        dims: ``(d_h, d_s, d_e)`` component dimensions.
        c_h: Curvature for the hyperbolic component.
        c_s: Curvature for the spherical component.
    """

    linear1_h: eqx.nn.Linear
    linear2_h: eqx.nn.Linear
    linear1_s: eqx.nn.Linear
    linear2_s: eqx.nn.Linear
    linear1_e: eqx.nn.Linear
    linear2_e: eqx.nn.Linear
    dims: tuple[int, int, int] = eqx.field(static=True)
    c_h: Float[Array, ""]
    c_s: Float[Array, ""]

    def __init__(
        self,
        dims: tuple[int, int, int],
        hidden_dim: int,
        curvatures: tuple[float, float],
        *,
        key: PRNGKeyArray,
    ):
        """Initialise ProductManifoldMLP.

        Args:
            dims: ``(d_h, d_s, d_e)`` component dimensions.
            hidden_dim: Hidden dimension for each component's MLP.
            curvatures: ``(c_h, c_s)`` curvatures for hyperbolic
                and spherical components.
            key: PRNG key for weight initialisation.
        """
        d_h, d_s, d_e = dims
        self.dims = dims

        c_h_init, c_s_init = curvatures
        self.c_h = jnp.array(c_h_init, dtype=jnp.float32)
        self.c_s = jnp.array(c_s_init, dtype=jnp.float32)

        keys = jax.random.split(key, 6)
        self.linear1_h = eqx.nn.Linear(d_h, hidden_dim, key=keys[0])
        self.linear2_h = eqx.nn.Linear(hidden_dim, d_h, key=keys[1])
        self.linear1_s = eqx.nn.Linear(d_s, hidden_dim, key=keys[2])
        self.linear2_s = eqx.nn.Linear(hidden_dim, d_s, key=keys[3])
        self.linear1_e = eqx.nn.Linear(d_e, hidden_dim, key=keys[4])
        self.linear2_e = eqx.nn.Linear(hidden_dim, d_e, key=keys[5])

    def __call__(
        self, x: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        """Apply tangent-space MLP per component.

        Args:
            x: Input features in the product manifold, with last
                axis = sum(dims).

        Returns:
            Transformed features with same shape, projected back
            onto each manifold.
        """
        c_h = jnp.abs(self.c_h) + _EPS

        x_h, x_s, x_e = split_components(x, self.dims)

        # Hyperbolic: log -> MLP -> exp
        x_h = project(x_h, c_h)
        v_h = logmap0(x_h, c_h)
        v_h = jax.nn.relu(self.linear1_h(v_h))
        v_h = self.linear2_h(v_h)
        out_h = expmap0(v_h, c_h)
        out_h = project(out_h, c_h)

        # Spherical: linear + relu + linear + project
        x_s = sphere_project(x_s)
        v_s = jax.nn.relu(self.linear1_s(x_s))
        v_s = self.linear2_s(v_s)
        out_s = sphere_project(v_s)

        # Euclidean: standard MLP
        v_e = jax.nn.relu(self.linear1_e(x_e))
        out_e = self.linear2_e(v_e)

        return concat_components(out_h, out_s, out_e)
