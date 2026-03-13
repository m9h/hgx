"""Neural ODE on a hypergraph: evolve node features in continuous time.

Demonstrates:
  1. Building a small hypergraph
  2. Wrapping a UniGCNConv as a HypergraphNeuralODE vector field
  3. Integrating the ODE and inspecting the trajectory
  4. Plotting feature norms over time (if matplotlib is available)

Usage::

    uv run python examples/neural_ode.py
"""

from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

import hgx


def main():
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # ------------------------------------------------------------------
    # 1. Build a hypergraph: 6 nodes, 3 hyperedges
    # ------------------------------------------------------------------
    #   e0 = {0, 1, 2}  (triangle)
    #   e1 = {2, 3, 4}  (overlapping)
    #   e2 = {4, 5}     (bridge)
    feat_dim = 8
    node_features = jax.random.normal(k1, (6, feat_dim))
    hg = hgx.from_edge_list(
        [(0, 1, 2), (2, 3, 4), (4, 5)],
        num_nodes=6,
        node_features=node_features,
    )
    print("Hypergraph")
    print(f"  Nodes: {hg.num_nodes}  Edges: {hg.num_edges}  "
          f"Feature dim: {feat_dim}")

    # ------------------------------------------------------------------
    # 2. Create a HypergraphNeuralODE
    # ------------------------------------------------------------------
    conv = hgx.UniGCNConv(in_dim=feat_dim, out_dim=feat_dim, key=k2)
    neural_ode = hgx.HypergraphNeuralODE(conv)
    print(f"\nNeural ODE: dx/dt = tanh(UniGCNConv(x(t), H))")

    # ------------------------------------------------------------------
    # 3. Integrate with intermediate snapshots
    # ------------------------------------------------------------------
    num_steps = 20
    ts = jnp.linspace(0.0, 1.0, num_steps + 1)
    saveat = diffrax.SaveAt(ts=ts)
    sol = neural_ode(hg, t0=0.0, t1=1.0, dt0=0.05, saveat=saveat)

    # sol.ys has shape (num_steps+1, 6, 8)
    trajectory = sol.ys
    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"  Initial feature norms: {jnp.linalg.norm(trajectory[0], axis=-1)}")
    print(f"  Final   feature norms: {jnp.linalg.norm(trajectory[-1], axis=-1)}")

    # ------------------------------------------------------------------
    # 4. Convenience wrapper: evolve -> Hypergraph
    # ------------------------------------------------------------------
    hg_evolved = hgx.evolve(neural_ode, hg, t0=0.0, t1=1.0)
    print(f"\nEvolved features match final snapshot: "
          f"{jnp.allclose(hg_evolved.node_features, trajectory[-1], atol=1e-4)}")

    # ------------------------------------------------------------------
    # 5. Gradients through the ODE
    # ------------------------------------------------------------------
    def loss_fn(model):
        sol = model(hg, t0=0.0, t1=1.0, dt0=0.05)
        return jnp.sum(sol.ys[-1] ** 2)

    grads = jax.grad(loss_fn)(neural_ode)
    grad_norm = jnp.linalg.norm(
        jax.flatten_util.ravel_pytree(eqx.filter(grads, eqx.is_array))[0]
    )
    print(f"Gradient norm through ODE: {grad_norm:.6f}")

    # ------------------------------------------------------------------
    # 6. Plot trajectory (optional)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        norms = jnp.linalg.norm(trajectory, axis=-1)  # (T, 6)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(6):
            ax.plot(ts, norms[:, i], label=f"node {i}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature norm")
        ax.set_title("HypergraphNeuralODE: node feature evolution")
        ax.legend(fontsize=8)
        fig.tight_layout()

        out_path = Path(__file__).parent / "neural_ode_trajectory.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nTrajectory plot saved to {out_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
