"""Neural Developmental Programs (NDP) on hypergraphs.

Implements biologically-inspired developmental programs where each node
(cell) runs a shared neural program that decides state updates, cell
division, and connectivity formation based on local neighborhood
information gathered via hypergraph message passing.

All operations are JIT-compatible. Topology growth uses pre-allocated
masked arrays so that array shapes remain fixed throughout the
``jax.lax.scan`` developmental loop.

References:
    - Mordvintsev et al. (2020). "Growing Neural Cellular Automata."
      Distill.
    - Najarro et al. (2023). "Neural Developmental Programs." NeurIPS
      Workshop on Agent Learning in Open-Endedness.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._dynamic import preallocate
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# CellProgram: the shared "DNA" neural network
# ---------------------------------------------------------------------------


class CellProgram(eqx.Module):
    """Shared neural program executed by every node (cell).

    Takes local state concatenated with neighborhood aggregation as input,
    and outputs a state update vector, a scalar grow logit, and
    connectivity logits used to wire new daughter nodes.

    Attributes:
        state_mlp: MLP that maps concatenated (state, neighbor_agg) to
            a state update delta of the same dimension as the input state.
        grow_head: Linear layer that maps the MLP hidden representation
            to a scalar logit controlling cell division.
        connect_head: Linear layer that maps the MLP hidden representation
            to connectivity logits used for wiring new hyperedges.
    """

    state_mlp: eqx.nn.MLP
    grow_head: eqx.nn.Linear
    connect_head: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize CellProgram.

        Args:
            state_dim: Dimensionality of node state features.
            hidden_dim: Width of the hidden layers in the MLP.
            key: PRNG key for weight initialization.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        # Input is (state || neighbor_agg), both of dim state_dim
        self.state_mlp = eqx.nn.MLP(
            in_size=state_dim * 2,
            out_size=state_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.relu,
            key=k1,
        )
        self.grow_head = eqx.nn.Linear(state_dim, 1, key=k2)
        self.connect_head = eqx.nn.Linear(state_dim, hidden_dim, key=k3)

    def __call__(
        self,
        local_state: Float[Array, " d"],
        neighbor_agg: Float[Array, " d"],
    ) -> tuple[Float[Array, " d"], Float[Array, ""], Float[Array, " h"]]:
        """Run the cell program on a single node.

        Args:
            local_state: Node feature vector of shape (state_dim,).
            neighbor_agg: Aggregated neighborhood information of shape
                (state_dim,).

        Returns:
            Tuple of (state_update, grow_logit, connect_logits):
                - state_update: additive delta for the node state, shape (d,).
                - grow_logit: scalar logit for cell division decision.
                - connect_logits: logits for connectivity, shape (hidden_dim,).
        """
        x = jnp.concatenate([local_state, neighbor_agg])
        state_update = self.state_mlp(x)
        grow_logit = self.grow_head(state_update).squeeze(-1)
        connect_logits = self.connect_head(state_update)
        return state_update, grow_logit, connect_logits


# ---------------------------------------------------------------------------
# HypergraphNDP: the full developmental program
# ---------------------------------------------------------------------------


class HypergraphNDP(eqx.Module):
    """Neural Developmental Program on a hypergraph.

    Each developmental step:
        1. Message passing (conv layer) aggregates neighborhood info.
        2. Every active node runs the shared ``CellProgram`` to produce
           a state update, a grow logit, and connectivity logits.
        3. State updates are applied additively to node features.
        4. Nodes whose ``sigmoid(grow_logit) > growth_threshold`` divide:
           a daughter node is birthed in the next available masked slot
           with inherited (noisy) features, connected to the parent's
           hyperedges.

    The hypergraph must be pre-allocated to ``(max_nodes, max_edges)``
    before development so that all arrays have fixed shapes throughout
    the ``jax.lax.scan`` loop.

    Attributes:
        program: Shared CellProgram (the "DNA").
        conv: Hypergraph convolution layer for message passing.
        growth_threshold: Sigmoid threshold above which a node divides.
        max_nodes: Pre-allocated node capacity (static).
        max_edges: Pre-allocated hyperedge capacity (static).
    """

    program: CellProgram
    conv: eqx.Module
    growth_threshold: float
    max_nodes: int = eqx.field(static=True)
    max_edges: int = eqx.field(static=True)

    def __init__(
        self,
        program: CellProgram,
        conv: eqx.Module,
        max_nodes: int,
        max_edges: int,
        growth_threshold: float = 0.5,
    ):
        """Initialize HypergraphNDP.

        Args:
            program: Shared CellProgram executed by every node.
            conv: Any hgx convolution layer (e.g. UniGCNConv) used for
                neighborhood message passing. Must map
                ``(max_nodes, d) -> (max_nodes, d)`` (preserving dim).
            max_nodes: Maximum node capacity for pre-allocation.
            max_edges: Maximum hyperedge capacity for pre-allocation.
            growth_threshold: Sigmoid threshold for cell division.
                Defaults to 0.5.
        """
        self.program = program
        self.conv = conv
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.growth_threshold = growth_threshold

    def __call__(
        self,
        hg: Hypergraph,
        *,
        key: PRNGKeyArray,
    ) -> Hypergraph:
        """Execute one developmental step.

        Args:
            hg: Input hypergraph (must be pre-allocated with masks).
            key: PRNG key for stochastic growth noise.

        Returns:
            Updated Hypergraph after one developmental step.
        """
        k_noise, k_grow = jax.random.split(key)

        # 1. Message passing to get neighborhood aggregation
        neighbor_agg = self.conv(hg)  # pyright: ignore[reportCallIssue] # (max_nodes, d)

        # 2. Run CellProgram on every node (vmapped)
        state_updates, grow_logits, connect_logits = jax.vmap(self.program)(
            hg.node_features, neighbor_agg
        )
        # state_updates: (max_nodes, d)
        # grow_logits: (max_nodes,)
        # connect_logits: (max_nodes, hidden_dim)

        # 3. Apply state updates (masked: only active nodes change)
        import typing
        node_mask = typing.cast(Array, hg.node_mask)  # (max_nodes,)
        new_features = hg.node_features + state_updates * node_mask[:, None]

        # 4. Growth: determine which active nodes want to divide
        grow_probs = jax.nn.sigmoid(grow_logits)  # (max_nodes,)
        wants_to_grow = (grow_probs > self.growth_threshold) & node_mask

        # Straight-through estimator for differentiability:
        # In the forward pass we use the hard threshold; in the backward
        # pass gradients flow through the sigmoid.
        wants_to_grow_st = grow_probs + jax.lax.stop_gradient(
            wants_to_grow.astype(jnp.float32) - grow_probs
        )

        # Available (inactive) node slots
        available = ~node_mask  # (max_nodes,)

        # Use scan to sequentially assign daughter nodes to free slots.
        # For each parent that wants to grow, if a slot is free, birth
        # a daughter node with noisy parent features.
        noise = jax.random.normal(k_noise, new_features.shape) * 0.01

        def _grow_step(carry, parent_idx):
            """Try to birth one daughter from parent_idx."""
            feats, inc, n_mask, e_mask, avail = carry

            parent_active = wants_to_grow_st[parent_idx]
            has_slot = jnp.any(avail)
            do_birth = (parent_active > 0.5) & has_slot

            # Pick first available slot
            # When no slot is available, argmax returns 0 but do_birth=False
            daughter_idx = jnp.argmax(avail.astype(jnp.int32))

            # Daughter features: parent features + small noise
            daughter_feats = feats[parent_idx] + noise[daughter_idx]

            # Daughter inherits parent's hyperedge memberships
            daughter_edges = inc[parent_idx]

            # Conditionally write (using where to stay JIT-compatible)
            feats = jnp.where(
                do_birth,
                feats.at[daughter_idx].set(daughter_feats),
                feats,
            )
            inc = jnp.where(
                do_birth,
                inc.at[daughter_idx].set(daughter_edges),
                inc,
            )
            n_mask = jnp.where(
                do_birth,
                n_mask.at[daughter_idx].set(True),
                n_mask,
            )
            avail = jnp.where(
                do_birth,
                avail.at[daughter_idx].set(False),
                avail,
            )

            return (feats, inc, n_mask, e_mask, avail), None

        parent_indices = jnp.arange(self.max_nodes)
        init_carry = (new_features, hg.incidence, node_mask, hg.edge_mask, available)
        (new_features, new_incidence, new_node_mask, new_edge_mask, _), _ = (
            jax.lax.scan(_grow_step, init_carry, parent_indices)  # pyright: ignore[reportCallIssue]
        )

        return Hypergraph(
            node_features=new_features,
            incidence=new_incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=new_node_mask,
            edge_mask=new_edge_mask,
            geometry=hg.geometry,
        )

    def develop(
        self,
        hg: Hypergraph,
        num_steps: int,
        *,
        key: PRNGKeyArray,
    ) -> Hypergraph:
        """Run multiple developmental steps via ``jax.lax.scan``.

        The input hypergraph is pre-allocated to ``(max_nodes, max_edges)``
        if it is not already at the correct capacity.

        Args:
            hg: Input hypergraph.
            num_steps: Number of developmental steps to execute.
            key: PRNG key (split across steps).

        Returns:
            Hypergraph after ``num_steps`` developmental steps.
        """
        # Pre-allocate if needed
        if (
            hg.node_features.shape[0] != self.max_nodes
            or hg.incidence.shape[1] != self.max_edges
        ):
            hg = preallocate(hg, self.max_nodes, self.max_edges)

        keys = jax.random.split(key, num_steps)

        def _step(hg, step_key):
            hg = self(hg, key=step_key)
            return hg, hg.node_features

        hg_final, _features_trajectory = jax.lax.scan(_step, hg, keys)  # pyright: ignore[reportCallIssue]
        return hg_final


# ---------------------------------------------------------------------------
# Functional helper: trajectory extraction
# ---------------------------------------------------------------------------


def develop_trajectory(
    ndp: HypergraphNDP,
    hg: Hypergraph,
    num_steps: int,
    *,
    key: PRNGKeyArray,
) -> tuple[Float[Array, " T"], Float[Array, "T n d"]]:
    """Run NDP development and return the feature trajectory.

    Convenience function that executes ``num_steps`` developmental steps
    and collects node features at each step, analogous to
    ``hgx.trajectory`` for continuous-time dynamics.

    Args:
        ndp: The Neural Developmental Program module.
        hg: Input hypergraph.
        num_steps: Number of developmental steps.
        key: PRNG key.

    Returns:
        Tuple ``(ts, features)`` where:
            - *ts* has shape ``(num_steps,)`` with integer step indices
              cast to float (0.0, 1.0, ..., num_steps - 1).
            - *features* has shape ``(num_steps, max_nodes, d)`` containing
              node features at each developmental step.
    """
    # Pre-allocate if needed
    if (
        hg.node_features.shape[0] != ndp.max_nodes
        or hg.incidence.shape[1] != ndp.max_edges
    ):
        hg = preallocate(hg, ndp.max_nodes, ndp.max_edges)

    keys = jax.random.split(key, num_steps)

    def _step(hg, step_key):
        hg = ndp(hg, key=step_key)
        return hg, hg.node_features

    _, features = jax.lax.scan(_step, hg, keys)  # pyright: ignore[reportCallIssue]
    ts = jnp.arange(num_steps, dtype=jnp.float32)
    return ts, features
