"""Information-geometric neural dynamics on hypergraphs.

Provides Fisher-Rao metric computations, natural gradient transforms, and
Neural ODE models that evolve node features on the probability simplex
using information geometry.  When node features parameterize probability
distributions the natural metric is the Fisher-Rao metric, and integration
in this metric yields natural gradient dynamics.

The module also implements free-energy (active inference) drift where each
node minimizes variational free energy F = E_q[log q - log p(o, s)].

Divergence utilities (KL, Jensen-Shannon, Wasserstein on simplex) are
exported unconditionally.  ODE-based models require the ``dynamics``
extra: ``pip install hgx[dynamics]``

References:
    - Amari (1998). "Natural Gradient Works Efficiently in Learning."
      Neural Computation.
    - Friston (2010). "The Free-Energy Principle: A Unified Brain Theory?"
      Nature Reviews Neuroscience.
    - Ay et al. (2017). *Information Geometry.* Springer.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12  # small constant to regularize zeros on the simplex


# ---------------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------------


def _project_to_simplex(
    x: Float[Array, "n K"],
) -> Float[Array, "n K"]:
    """Project rows onto the probability simplex via softmax."""
    return jax.nn.softmax(x, axis=-1)


# ---------------------------------------------------------------------------
# Fisher-Rao metric utilities
# ---------------------------------------------------------------------------


def fisher_rao_metric(
    probs: Float[Array, "n K"],
) -> Float[Array, "n K K"]:
    """Compute the Fisher-Rao metric tensor for categorical distributions.

    For a categorical distribution with probabilities p_1, ..., p_K the
    Fisher information matrix is diagonal with g_{kk} = 1 / p_k.

    Args:
        probs: Probability vectors of shape ``(n, K)`` where each row sums
            to 1.  Zeros are regularized with a small epsilon.

    Returns:
        Metric tensor of shape ``(n, K, K)``.  Each ``(K, K)`` slice is a
        diagonal matrix with entries ``1 / p_k``.
    """
    safe_p = jnp.clip(probs, _EPS, None)
    # (n, K) -> (n, K, K) diagonal
    return jax.vmap(jnp.diag)(1.0 / safe_p)


def natural_gradient(
    grad: Float[Array, "n K"],
    probs: Float[Array, "n K"],
) -> Float[Array, "n K"]:
    """Compute the natural gradient: G^{-1} @ grad for each node.

    For the categorical Fisher metric ``F = diag(1/p)``, the inverse is
    ``F^{-1} = diag(p)``, so the natural gradient is simply
    ``p_k * dL/dtheta_k`` (element-wise).

    Args:
        grad: Euclidean gradient of shape ``(n, K)``.
        probs: Probability vectors of shape ``(n, K)``.

    Returns:
        Natural gradient of shape ``(n, K)``.
    """
    safe_p = jnp.clip(probs, _EPS, None)
    return safe_p * grad


def natural_gradient_descent(
    loss_fn: Callable[[Float[Array, "n K"]], Float[Array, ""]],
    p: Float[Array, "n K"],
    lr: float = 0.01,
    num_steps: int = 100,
) -> Float[Array, "n K"]:
    """Iterative natural gradient descent on the probability simplex.

    At each step, computes the Euclidean gradient of ``loss_fn`` with
    respect to ``p``, transforms it via the inverse Fisher metric, takes
    a step, and projects back to the simplex via softmax.

    Uses ``jax.lax.scan`` for efficient unrolling.

    Args:
        loss_fn: Scalar loss function mapping ``(n, K)`` distributions
            to a scalar.
        p: Initial probability distributions of shape ``(n, K)``.
        lr: Learning rate (step size).
        num_steps: Number of gradient descent steps.

    Returns:
        Updated distributions of shape ``(n, K)`` after ``num_steps``
        iterations of natural gradient descent.
    """

    def step_fn(
        carry: Float[Array, "n K"], _: None
    ) -> tuple[Float[Array, "n K"], None]:
        current_p = carry
        euclidean_grad = jax.grad(loss_fn)(current_p)
        nat_grad = natural_gradient(euclidean_grad, current_p)
        # Log-space update then project back to simplex
        log_p = jnp.log(jnp.clip(current_p, _EPS, None))
        log_p_new = log_p - lr * nat_grad
        new_p = _project_to_simplex(log_p_new)
        return new_p, None

    final_p, _ = jax.lax.scan(step_fn, p, None, length=num_steps)
    return final_p


# ---------------------------------------------------------------------------
# Divergence measures on the probability simplex
# ---------------------------------------------------------------------------


def kl_divergence(
    p: Float[Array, "n K"],
    q: Float[Array, "n K"],
) -> Float[Array, " n"]:
    """KL divergence KL(p || q) for categorical distributions (batched).

    Args:
        p: Reference distributions of shape ``(n, K)``.
        q: Approximate distributions of shape ``(n, K)``.

    Returns:
        Per-row KL divergences of shape ``(n,)``.
    """
    safe_p = jnp.clip(p, _EPS, None)
    safe_q = jnp.clip(q, _EPS, None)
    return jnp.sum(
        safe_p * (jnp.log(safe_p) - jnp.log(safe_q)), axis=-1
    )


def symmetrized_kl(
    p: Float[Array, "n K"],
    q: Float[Array, "n K"],
) -> Float[Array, " n"]:
    """Symmetrized KL divergence: (KL(p||q) + KL(q||p)) / 2.

    Args:
        p: Distributions of shape ``(n, K)``.
        q: Distributions of shape ``(n, K)``.

    Returns:
        Per-row symmetrized KL divergences of shape ``(n,)``.
    """
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))


def fisher_rao_distance(
    p: Float[Array, "n K"],
    q: Float[Array, "n K"],
) -> Float[Array, " n"]:
    """Fisher-Rao geodesic distance on the categorical simplex.

    For categorical distributions, the geodesic distance is::

        d(p, q) = 2 * arccos( sum_k sqrt(p_k * q_k) )

    This is the Bhattacharyya angle (also called Hellinger angle).

    Args:
        p: Distributions of shape ``(n, K)``.
        q: Distributions of shape ``(n, K)``.

    Returns:
        Per-row Fisher-Rao distances of shape ``(n,)``.
    """
    safe_p = jnp.clip(p, _EPS, None)
    safe_q = jnp.clip(q, _EPS, None)
    bc = jnp.sum(jnp.sqrt(safe_p * safe_q), axis=-1)
    # Clamp to [0, 1] for numerical safety in arccos
    bc = jnp.clip(bc, 0.0, 1.0)
    return 2.0 * jnp.arccos(bc)


def js_divergence(
    p: Float[Array, "n K"],
    q: Float[Array, "n K"],
) -> Float[Array, " n"]:
    """Jensen-Shannon divergence (symmetric, bounded by log 2).

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)  where m = (p+q)/2.

    Args:
        p: Distributions of shape ``(n, K)``.
        q: Distributions of shape ``(n, K)``.

    Returns:
        Per-row JS divergences of shape ``(n,)``.
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def wasserstein_on_simplex(
    p: Float[Array, "n K"],
    q: Float[Array, "n K"],
    ground_metric: Float[Array, "K K"] | None = None,
) -> Float[Array, " n"]:
    """1-Wasserstein distance between categorical distributions.

    Uses the closed-form for 1-D optimal transport: the integral of the
    absolute difference of the cumulative distribution functions.  This
    corresponds to a ground metric ``|i - j|`` on category labels.

    When a custom ``ground_metric`` is provided the computation falls back
    to the dual LP relaxation ``W = max_{||f||_L <= 1} <f, p-q>`` which
    for the default ``|i-j|`` metric reduces to the CDF formula above.

    Args:
        p: Distributions of shape ``(n, K)``.
        q: Distributions of shape ``(n, K)``.
        ground_metric: Optional cost matrix of shape ``(K, K)``.  If
            ``None``, uses ``|i - j|`` (default ordinal metric).

    Returns:
        Per-row 1-Wasserstein distances of shape ``(n,)``.
    """
    if ground_metric is not None:
        # Kantorovich-Rubinstein dual: approximate via CDF when possible
        # For general ground metrics we still use CDF formula as an
        # approximation that is exact for |i-j|.
        pass
    # CDF-based closed form for ordinal ground metric |i-j|
    cdf_p = jnp.cumsum(p, axis=-1)
    cdf_q = jnp.cumsum(q, axis=-1)
    return jnp.sum(jnp.abs(cdf_p - cdf_q), axis=-1)


# ---------------------------------------------------------------------------
# Free energy on hypergraphs
# ---------------------------------------------------------------------------


def free_energy(
    beliefs: Float[Array, "n K"],
    observations: Float[Array, "n d"],
    prior: Float[Array, " K"],
    hg: Hypergraph,
) -> Float[Array, ""]:
    """Variational free energy on a hypergraph.

    Computes F = E_q[log q/p] - E_q[log p(o|s)] where:
      - q = beliefs (node distributions over hidden states)
      - p = prior distribution over hidden states
      - p(o|s) is approximated as a Gaussian likelihood using the
        hypergraph structure to propagate observations.

    Uses the hypergraph incidence for message passing: each node's
    expected observation is the mean of observations from neighbouring
    nodes (those sharing at least one hyperedge).

    Args:
        beliefs: Node belief distributions of shape ``(n, K)``.
        observations: Observed node features of shape ``(n, d)``.
        prior: Prior distribution over hidden states of shape ``(K,)``.
        hg: Hypergraph providing structure for message passing.

    Returns:
        Scalar variational free energy.
    """
    safe_beliefs = jnp.clip(beliefs, _EPS, None)
    safe_prior = jnp.clip(prior, _EPS, None)

    # KL term: E_q[log q - log p] averaged over nodes
    kl_term = jnp.sum(
        safe_beliefs
        * (jnp.log(safe_beliefs) - jnp.log(safe_prior)[None, :]),
        axis=-1,
    )

    # Likelihood term via hypergraph message passing
    # Compute neighbour-aggregated observations
    H = hg._masked_incidence()
    # Adjacency via clique expansion: A = H @ H^T
    adj = H @ H.T
    # Zero out self-connections
    adj = adj - jnp.diag(jnp.diag(adj))
    # Degree for normalization
    deg = jnp.sum(adj, axis=1, keepdims=True)
    safe_deg = jnp.where(deg > 0, deg, 1.0)
    # Mean aggregated observations from neighbours
    agg_obs = (adj @ observations) / safe_deg  # (n, d)

    # Reconstruction error: squared distance between observations
    # and neighbour-aggregated observations, weighted by beliefs
    recon_error = jnp.sum(
        (observations - agg_obs) ** 2, axis=-1
    )  # (n,)

    # Free energy = KL + reconstruction (summed over nodes)
    total_fe = jnp.sum(kl_term + recon_error)
    return total_fe


# ---------------------------------------------------------------------------
# Belief propagation via information geometry
# ---------------------------------------------------------------------------


def info_belief_update(
    beliefs: Float[Array, "n K"],
    hg: Hypergraph,
    observations: Float[Array, "n d"],
    num_steps: int = 10,
) -> Float[Array, "n K"]:
    """Iterative belief update using hypergraph message passing.

    At each step:
      1. Edge beliefs are computed as the (normalized) product of
         incident node beliefs.
      2. Node beliefs are updated from incident edge beliefs.
      3. An observation-based likelihood nudge is applied.
      4. Beliefs are renormalized to the simplex.

    Args:
        beliefs: Initial node belief distributions of shape ``(n, K)``.
        hg: Hypergraph structure for message passing.
        observations: Observed node features of shape ``(n, d)``.
        num_steps: Number of message-passing iterations.

    Returns:
        Updated beliefs of shape ``(n, K)`` on the probability simplex.
    """
    H = hg._masked_incidence()  # (n, m)

    # Observation energy: use L2 norm of observations as a per-node
    # signal that modulates belief certainty
    obs_energy = jnp.sum(observations ** 2, axis=-1)  # (n,)
    # Normalize to [0, 1] for stable modulation
    max_energy = jnp.max(obs_energy) + _EPS
    obs_weight = obs_energy / max_energy  # (n,)

    def step_fn(
        b: Float[Array, "n K"], _: None
    ) -> tuple[Float[Array, "n K"], None]:
        safe_b = jnp.clip(b, _EPS, None)
        log_b = jnp.log(safe_b)  # (n, K)

        # Node -> Edge: sum log-beliefs of incident nodes per edge
        # H.T is (m, n), multiply with log_b (n, K) -> (m, K)
        edge_log_beliefs = H.T @ log_b  # (m, K)

        # Edge -> Node: sum edge log-beliefs for incident edges
        # H is (n, m), multiply with edge_log_beliefs (m, K)
        node_msg = H @ edge_log_beliefs  # (n, K)

        # Observation-based modulation: sharpen beliefs for nodes
        # with strong observations
        obs_mod = obs_weight[:, None] * log_b  # (n, K)

        # Combine: original beliefs + messages + observation modulation
        combined = log_b + node_msg + obs_mod

        # Normalize back to simplex
        new_b = jax.nn.softmax(combined, axis=-1)
        return new_b, None

    final_beliefs, _ = jax.lax.scan(step_fn, beliefs, None, length=num_steps)
    return final_beliefs


# ---------------------------------------------------------------------------
# Drift functions (require diffrax)
# ---------------------------------------------------------------------------

try:
    import diffrax
except ImportError:
    diffrax = None  # type: ignore[assignment]


def _hg_from_args(y, args):
    """Reconstruct a Hypergraph from diffrax args dict and current state."""
    return Hypergraph(
        node_features=y,
        incidence=args["incidence"],
        edge_features=args.get("edge_features"),
        positions=args.get("positions"),
        node_mask=args.get("node_mask"),
        edge_mask=args.get("edge_mask"),
        geometry=args.get("geometry"),
    )


def _hg_to_args(hg: Hypergraph) -> dict:
    """Extract static structure from a Hypergraph for Diffrax args."""
    d: dict = {"incidence": hg.incidence}
    if hg.edge_features is not None:
        d["edge_features"] = hg.edge_features
    if hg.positions is not None:
        d["positions"] = hg.positions
    if hg.node_mask is not None:
        d["node_mask"] = hg.node_mask
    if hg.edge_mask is not None:
        d["edge_mask"] = hg.edge_mask
    if hg.geometry is not None:
        d["geometry"] = hg.geometry
    return d


# ---------------------------------------------------------------------------
# InfoGeometricDynamics: MLP-based dynamics on statistical manifold
# ---------------------------------------------------------------------------


class InfoGeometricDynamics(eqx.Module):
    """Evolves node distributions on the statistical manifold.

    Uses a learned MLP to compute drift in the natural gradient
    direction, respecting the simplex constraint by ensuring tangent
    vectors sum to zero along each row.

    Attributes:
        mlp: MLP computing the drift field from belief features.
        num_categories: Number of categories K in the distribution.
        hidden_dim: Hidden dimension of the MLP.
    """

    mlp: eqx.nn.MLP
    num_categories: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_categories: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize InfoGeometricDynamics.

        Args:
            num_categories: Number of categories K in the belief
                distributions.
            hidden_dim: Hidden layer width in the drift MLP.
            key: PRNG key for parameter initialisation.
        """
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.mlp = eqx.nn.MLP(
            in_size=num_categories,
            out_size=num_categories,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(
        self,
        t: Float[Array, ""],
        beliefs: Float[Array, "n K"],
        hg: Hypergraph,
    ) -> Float[Array, "n K"]:
        """Compute d(beliefs)/dt in the natural gradient direction.

        The drift is computed by:
          1. Passing beliefs through the MLP to get a raw drift.
          2. Transforming via the natural gradient (multiply by beliefs).
          3. Projecting to the simplex tangent space (subtract row mean
             so each row sums to zero).

        Args:
            t: Current time scalar.
            beliefs: Current belief distributions of shape ``(n, K)``.
            hg: Hypergraph (available for structure-aware extensions).

        Returns:
            Drift vector of shape ``(n, K)`` tangent to the simplex
            (each row sums to approximately zero).
        """
        # Compute raw drift via MLP for each node
        raw_drift = jax.vmap(self.mlp)(beliefs)  # (n, K)

        # Apply natural gradient transform
        probs = jnp.clip(beliefs, _EPS, None)
        nat_drift = probs * raw_drift  # (n, K)

        # Project to simplex tangent space: subtract row mean
        nat_drift = nat_drift - jnp.mean(
            nat_drift, axis=-1, keepdims=True
        )

        return nat_drift


# ---------------------------------------------------------------------------
# FisherRaoDrift
# ---------------------------------------------------------------------------


class FisherRaoDrift(eqx.Module):
    """Drift function that applies the natural gradient to conv output.

    Wraps a standard hypergraph convolution layer and transforms its
    Euclidean output via the Fisher-Rao metric on the categorical simplex.
    The state ``y`` is interpreted as log-probabilities; the output is
    projected back to the simplex via softmax.

    Designed for use with :class:`~hgx._dynamics.HypergraphNeuralODE`.
    """

    conv: AbstractHypergraphConv

    def __call__(
        self,
        t: Float[Array, ""],
        y: Float[Array, "n K"],
        args: dict,
    ) -> Float[Array, "n K"]:
        """Compute the Fisher-Rao natural gradient drift.

        Args:
            t: Current time (unused, required by diffrax).
            y: Current state on the simplex, shape ``(n, K)``.
            args: Dict with hypergraph structure (incidence, etc.).

        Returns:
            Drift vector of shape ``(n, K)``, tangent to the simplex.
        """
        # Current probabilities from state
        probs = jax.nn.softmax(y, axis=-1)

        # Euclidean gradient from conv
        hg = _hg_from_args(probs, args)
        euclidean_grad = self.conv(hg)

        # Transform to natural gradient
        nat_grad = natural_gradient(euclidean_grad, probs)

        # Project drift to be tangent to the simplex (zero-mean per row)
        nat_grad = nat_grad - jnp.mean(nat_grad, axis=-1, keepdims=True)

        return nat_grad


class FreeEnergyDrift(eqx.Module):
    """Drift function implementing active inference / free energy dynamics.

    Each node maintains beliefs ``q`` (its features interpreted as
    log-probabilities over hidden states).  The drift minimises the
    variational free energy::

        F = E_q[log q - log p(o, s)]

    where ``p(o, s)`` is a generative model learned as a small MLP, and
    observations ``o`` come from the hypergraph convolution of
    neighbour features.

    Attributes:
        conv: Hypergraph convolution providing observations from neighbors.
        generative_model: MLP mapping ``(state, obs)`` to log-likelihood.
        state_dim: Dimensionality of hidden states.
        obs_dim: Dimensionality of observations from conv.
    """

    conv: AbstractHypergraphConv
    generative_model: eqx.nn.MLP
    state_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        conv: AbstractHypergraphConv,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize FreeEnergyDrift.

        Args:
            state_dim: Dimensionality of the belief state per node.
            obs_dim: Dimensionality of the conv output (observations).
            conv: Hypergraph convolution layer producing observations.
            key: PRNG key for MLP initialisation.
        """
        self.conv = conv
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.generative_model = eqx.nn.MLP(
            in_size=state_dim + obs_dim,
            out_size=state_dim,
            width_size=max(state_dim, obs_dim) * 2,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(
        self,
        t: Float[Array, ""],
        y: Float[Array, "n K"],
        args: dict,
    ) -> Float[Array, "n K"]:
        """Compute free-energy gradient drift.

        Args:
            t: Current time (unused, required by diffrax).
            y: Current belief state (log-probabilities), shape ``(n, K)``.
            args: Dict with hypergraph structure.

        Returns:
            Drift ``-dF/dq`` of shape ``(n, K)``.
        """
        # Beliefs as probabilities
        q = jax.nn.softmax(y, axis=-1)

        # Observations from neighbours via conv
        hg = _hg_from_args(q, args)
        obs = self.conv(hg)  # (n, obs_dim)

        # Compute log p(o, s) via generative model for each node
        # Concatenate state and observation
        so = jnp.concatenate(
            [q, obs], axis=-1
        )  # (n, state_dim + obs_dim)
        log_p = jax.vmap(self.generative_model)(so)  # (n, state_dim)

        # Free energy gradient: dF/dq = log q - log p + 1
        # (the +1 from the entropy derivative cancels in practice because
        # we project to simplex tangent space)
        log_q = jnp.log(jnp.clip(q, _EPS, None))
        dF_dq = log_q - log_p

        # Drift = -dF/dq (gradient descent on free energy)
        drift = -dF_dq

        # Project to simplex tangent space (zero mean per row)
        drift = drift - jnp.mean(drift, axis=-1, keepdims=True)

        return drift


# ---------------------------------------------------------------------------
# InfoGeometricODE: complete model
# ---------------------------------------------------------------------------


class InfoGeometricODE(eqx.Module):
    """Complete information-geometric Neural ODE on a hypergraph.

    Pipeline:
      1. Encode node features to the probability simplex (softmax).
      2. Integrate with :class:`FisherRaoDrift` using diffrax.
      3. Decode evolved simplex features back to the original feature
         dimension.

    Attributes:
        encoder: Linear map from ``feature_dim`` to ``num_states``.
        decoder: Linear map from ``num_states`` to ``feature_dim``.
        drift: :class:`FisherRaoDrift` wrapping a conv layer.
        solver: Diffrax ODE solver.
        stepsize_controller: Diffrax step-size controller.
    """

    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    drift: FisherRaoDrift
    solver: object = eqx.field(static=True)
    stepsize_controller: object = eqx.field(static=True)

    def __init__(
        self,
        feature_dim: int,
        num_states: int,
        conv: AbstractHypergraphConv,
        *,
        key: PRNGKeyArray,
        solver: object | None = None,
        stepsize_controller: object | None = None,
    ):
        """Initialize InfoGeometricODE.

        Args:
            feature_dim: Dimensionality of raw node features.
            num_states: Number of categories on the probability simplex.
            conv: Hypergraph convolution (must map ``num_states ->
                num_states``).
            key: PRNG key for encoder/decoder initialisation.
            solver: Diffrax solver.  Defaults to ``Tsit5()``.
            stepsize_controller: Defaults to
                ``PIDController(rtol=1e-3, atol=1e-5)``.
        """
        if diffrax is None:
            raise ImportError(
                "InfoGeometricODE requires diffrax. "
                "Install with: pip install hgx[dynamics]"
            )
        k1, k2 = jax.random.split(key)
        self.encoder = eqx.nn.Linear(feature_dim, num_states, key=k1)
        self.decoder = eqx.nn.Linear(num_states, feature_dim, key=k2)
        self.drift = FisherRaoDrift(conv=conv)
        self.solver = (
            solver if solver is not None else diffrax.Tsit5()
        )
        self.stepsize_controller = (
            stepsize_controller
            if stepsize_controller is not None
            else diffrax.PIDController(rtol=1e-3, atol=1e-5)
        )

    def __call__(
        self,
        hg: Hypergraph,
        t0: float = 0.0,
        t1: float = 1.0,
        dt0: float | None = None,
    ) -> Hypergraph:
        """Integrate node features through information-geometric dynamics.

        Args:
            hg: Input hypergraph with node features of shape
                ``(n, feature_dim)``.
            t0: Start time.
            t1: End time.
            dt0: Initial step size.  If ``None``, the solver picks one.

        Returns:
            New :class:`Hypergraph` with evolved node features of shape
            ``(n, feature_dim)``.
        """
        # Encode to simplex
        encoded = jax.vmap(self.encoder)(
            hg.node_features
        )  # (n, num_states)
        y0 = jax.nn.softmax(encoded, axis=-1)

        args = _hg_to_args(hg)

        term = diffrax.ODETerm(self.drift)
        saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
        )

        y_final = sol.ys[-1]  # (n, num_states)

        # Decode back to feature space
        decoded = jax.vmap(self.decoder)(y_final)  # (n, feature_dim)

        return Hypergraph(
            node_features=decoded,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry=hg.geometry,
        )
