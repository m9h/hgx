"""PGMax bridge: factor-graph inference on hypergraph topology.

Requires optional dependency: ``pip install hgx[pgmax]``
(pgmax and its dependencies including numba).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hgx._hypergraph import Hypergraph


if TYPE_CHECKING:
    from pgmax.fgraph.fgraph import FactorGraph
    from pgmax.vgroup.varray import NDVarArray


def _ensure_pgmax():
    """Lazily import and return pgmax submodules.

    Returns a namespace-like object with ``vgroup``, ``fgroup``, ``fgraph``
    attributes.
    """
    try:
        from pgmax import fgraph, fgroup, vgroup
    except ImportError as exc:
        raise ImportError(
            "PGMax bridge requires pgmax. "
            "Install with: pip install hgx[pgmax]"
        ) from exc

    class _PGMax:
        pass

    ns = _PGMax()
    ns.vgroup = vgroup
    ns.fgroup = fgroup
    ns.fgraph = fgraph
    return ns


def hypergraph_to_factor_graph(
    hg: Hypergraph,
    num_states: int,
    potential_fn: Any | None = None,
) -> tuple[FactorGraph, NDVarArray]:
    """Convert a Hypergraph to a PGMax FactorGraph.

    Each node becomes a discrete variable with ``num_states`` states.
    Each hyperedge with >= 2 members becomes an EnumFactor connecting
    its member nodes, with all state configurations enumerated.

    Hyperedges are grouped by size so that those with the same number
    of members share an ``EnumFactorGroup`` for efficiency.

    Note: The number of configurations per factor grows as
    ``num_states ** hyperedge_size``. Keep hyperedges small or use
    few states to avoid memory issues.

    Args:
        hg: Input hypergraph.
        num_states: Number of discrete states per variable.
        potential_fn: Optional callable ``(config, member_indices) -> float``
            returning the log potential for a given configuration array
            and numpy array of member node indices. If *None*, uniform
            potentials (all zeros) are used.

    Returns:
        Tuple of ``(factor_graph, variables)`` where *variables* is the
        ``NDVarArray`` needed to set evidence and read beliefs.
    """
    pgmax = _ensure_pgmax()

    H = np.asarray(hg._masked_incidence())
    n, m = H.shape

    variables = pgmax.vgroup.NDVarArray(num_states=num_states, shape=(n,))
    fg = pgmax.fgraph.FactorGraph(variable_groups=variables)

    # Group hyperedges by member count for efficient EnumFactorGroup creation
    size_to_members: dict[int, list[np.ndarray]] = defaultdict(list)
    for k in range(m):
        members = np.where(H[:, k] > 0)[0]
        if len(members) >= 2:
            size_to_members[len(members)].append(members)

    for size, members_list in size_to_members.items():
        grids = np.meshgrid(*[np.arange(num_states)] * size, indexing="ij")
        configs = np.stack([g.ravel() for g in grids], axis=-1).astype(np.int32)

        variables_for_factors = [
            [variables[int(i)] for i in members] for members in members_list
        ]

        if potential_fn is not None:
            log_pots = np.array(
                [
                    [float(potential_fn(cfg, members)) for cfg in configs]
                    for members in members_list
                ],
                dtype=np.float64,
            )
        else:
            log_pots = None

        factor_group = pgmax.fgroup.EnumFactorGroup(
            variables_for_factors=variables_for_factors,
            factor_configs=configs,
            log_potentials=log_pots,
        )
        fg.add_factors(factor_group)

    return fg, variables


def learn_potentials(
    hg: Hypergraph,
    conv: Any,
    num_states: int,
    *,
    key: jax.Array,
) -> jnp.ndarray:
    """Derive per-node log-probabilities from a conv layer.

    Runs the conv layer to get embeddings, applies ``log_softmax``, and
    returns an array suitable for use as BP evidence via
    ``bp.init(evidence_updates={variables: result})``.

    The conv layer's output dimension must be >= ``num_states``.

    Args:
        hg: Input hypergraph with node features.
        conv: An hgx conv layer producing ``(n, out_dim)`` output.
        num_states: Number of discrete states.
        key: PRNG key (reserved for future stochastic layers).

    Returns:
        Array of shape ``(n, num_states)`` with log-probabilities.
    """
    logits = conv(hg)  # (n, out_dim)
    return jax.nn.log_softmax(logits[:, :num_states], axis=-1)


def run_cell_fate_inference(
    hg: Hypergraph,
    num_fates: int,
    conv: Any | None = None,
    num_bp_iters: int = 50,
    *,
    key: jax.Array | None = None,
) -> dict[str, Any]:
    """End-to-end cell fate inference via belief propagation.

    Builds a factor graph from the hypergraph topology, optionally
    computes node-level evidence from a conv layer, runs loopy BP,
    and returns marginal beliefs for each node.

    Args:
        hg: Input hypergraph (nodes = cells, hyperedges = cell groups).
        num_fates: Number of discrete cell fates.
        conv: Optional conv layer for computing evidence. Output
            dimension must be >= ``num_fates``.
        num_bp_iters: Number of belief propagation iterations.
        key: PRNG key (required if *conv* is provided).

    Returns:
        Dictionary with keys:

        - ``"marginals"``: Array of shape ``(n, num_fates)`` with
          marginal probabilities for each cell fate.
        - ``"map_states"``: Array of shape ``(n,)`` with MAP state
          assignments.
        - ``"beliefs"``: Raw belief dict from PGMax.
        - ``"factor_graph"``: The constructed PGMax FactorGraph.
    """
    _ensure_pgmax()
    from pgmax.infer import bp as pgmax_bp

    fg, variables = hypergraph_to_factor_graph(hg, num_fates)

    bp_state = fg.bp_state
    bp = pgmax_bp.BP(bp_state, temperature=0.0)

    if conv is not None:
        if key is None:
            raise ValueError("Must provide key when using a conv layer.")
        evidence = learn_potentials(hg, conv, num_fates, key=key)
        bp_arrays = bp.init(
            evidence_updates={variables: np.asarray(evidence)},
        )
    else:
        bp_arrays = bp.init()

    bp_arrays = bp.run(bp_arrays, num_iters=num_bp_iters, damping=0.5)

    beliefs = bp.get_beliefs(bp_arrays)
    marginals = pgmax_bp.get_marginals(beliefs)

    marginals_arr = np.asarray(marginals[variables])
    map_states = np.argmax(marginals_arr, axis=-1)

    return {
        "marginals": marginals_arr,
        "map_states": map_states,
        "beliefs": beliefs,
        "factor_graph": fg,
    }


class ActiveInferenceStep(eqx.Module):
    """Combines continuous Neural ODE dynamics with PGMax belief propagation.

    At each step:

    1. Evolve node features forward in time via a continuous dynamics model.
    2. Map evolved features to discrete state evidence via a conv layer.
    3. Run belief propagation on the hypergraph factor graph.
    4. Return marginal beliefs over discrete states.

    Attributes:
        ode_model: A continuous dynamics model (e.g. HypergraphNeuralODE).
        conv: Conv layer mapping node features to ``(n, num_states)`` logits.
        num_states: Number of discrete states per node.
        num_bp_iters: Number of BP iterations per step.
    """

    ode_model: Any
    conv: Any
    num_states: int = eqx.field(static=True)
    num_bp_iters: int = eqx.field(static=True, default=50)

    def __call__(
        self,
        hg: Hypergraph,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        key: jax.Array,
    ) -> dict[str, Any]:
        """Run one active inference step.

        Args:
            hg: Input hypergraph with current node features.
            t0: Start time for ODE integration.
            t1: End time for ODE integration.
            key: PRNG key.

        Returns:
            Dictionary with keys:

            - ``"marginals"``: Marginal beliefs ``(n, num_states)``.
            - ``"map_states"``: MAP state assignments ``(n,)``.
            - ``"evolved_hg"``: Hypergraph with evolved node features.
        """
        # Step 1: Evolve continuous dynamics
        sol = self.ode_model(hg, t0=t0, t1=t1)
        evolved_features = sol.ys

        evolved_hg = Hypergraph(
            node_features=evolved_features,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry=hg.geometry,
        )

        # Step 2 + 3: Discrete inference with conv-derived evidence
        result = run_cell_fate_inference(
            evolved_hg,
            self.num_states,
            conv=self.conv,
            num_bp_iters=self.num_bp_iters,
            key=key,
        )
        result["evolved_hg"] = evolved_hg
        return result
