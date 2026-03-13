"""Dynamic topology: grow a hypergraph and run convolutions at each step.

Demonstrates the preallocate → add_node → conv → add_hyperedge workflow.
All operations are JIT-compatible (array shapes never change).

Usage::

    uv run python examples/dynamic_topology.py
"""

import jax
import jax.numpy as jnp

import hgx


def main():
    key = jax.random.PRNGKey(0)
    feat_dim = 8

    # --- Start with a small hypergraph: 3 nodes, 2 hyperedges ---
    # e0 = {0, 1}, e1 = {1, 2}
    hg = hgx.from_edge_list(
        [(0, 1), (1, 2)],
        num_nodes=3,
        node_features=jax.random.normal(key, (3, feat_dim)),
    )
    print("Initial hypergraph")
    print(f"  Nodes: {hg.num_nodes}  Edges: {hg.num_edges}")
    print(f"  Incidence:\n{hg.incidence}\n")

    # --- Pre-allocate capacity for growth ---
    hg = hgx.preallocate(hg, max_nodes=6, max_edges=5)
    print("After preallocate(max_nodes=6, max_edges=5)")
    print(f"  Active nodes: {hg.num_nodes}  Active edges: {hg.num_edges}")
    print(f"  Allocated: {hg.max_nodes} node slots, {hg.max_edges} edge slots\n")

    # --- Run a convolution on the initial topology ---
    k1, k2, k3 = jax.random.split(key, 3)
    conv = hgx.UniGCNConv(in_dim=feat_dim, out_dim=feat_dim, key=k1)
    out = conv(hg)
    print("Conv on initial topology")
    print(f"  Output shape: {out.shape}")
    print(f"  Masked node 3 output (should be zeros): {out[3]}\n")

    # --- Add a new node connected to hyperedge 0 ---
    new_features = jax.random.normal(k2, (feat_dim,))
    membership = jnp.array([True, False, False, False, False])  # max_edges=5
    hg = hgx.add_node(hg, features=new_features, hyperedges=membership)
    print("After add_node (connected to e0)")
    print(f"  Active nodes: {hg.num_nodes}  Active edges: {hg.num_edges}")
    print(f"  Node mask: {hg.node_mask}")

    # Run conv again — new node participates in message passing
    out = conv(hg)
    print(f"  Conv output shape: {out.shape}")
    print(f"  New node (idx 3) output: {out[3]}\n")

    # --- Add another node ---
    new_features_2 = jax.random.normal(k3, (feat_dim,))
    membership_2 = jnp.array([False, True, False, False, False])  # edge 1
    hg = hgx.add_node(hg, features=new_features_2, hyperedges=membership_2)
    print("After second add_node (connected to e1)")
    print(f"  Active nodes: {hg.num_nodes}  Active edges: {hg.num_edges}\n")

    # --- Add a new hyperedge connecting original node 0 and new nodes 3, 4 ---
    members = jnp.array([True, False, False, True, True, False])  # max_nodes=6
    hg = hgx.add_hyperedge(hg, members=members)
    print("After add_hyperedge({0, 3, 4})")
    print(f"  Active nodes: {hg.num_nodes}  Active edges: {hg.num_edges}")
    print(f"  Edge mask: {hg.edge_mask}")

    # Run conv on the grown topology
    out = conv(hg)
    print(f"  Conv output shape: {out.shape}\n")

    # --- Remove a node and observe ---
    hg = hgx.remove_node(hg, idx=1)
    print("After remove_node(1)")
    print(f"  Active nodes: {hg.num_nodes}  Active edges: {hg.num_edges}")
    print(f"  Node mask: {hg.node_mask}")

    out = conv(hg)
    print(f"  Removed node output (should be zeros): {out[1]}\n")

    # --- Verify JIT compatibility ---
    import equinox as eqx
    jit_conv = eqx.filter_jit(conv)
    out_jit = jit_conv(hg)
    match = jnp.allclose(out, out_jit, atol=1e-6)
    print(f"JIT output matches eager: {match}")

    print("\nDone — all dynamic topology operations succeeded.")


if __name__ == "__main__":
    main()
