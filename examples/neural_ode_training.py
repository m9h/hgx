"""Neural ODE training on a hypergraph — canonical example.

Demonstrates how to:
  1. Build a HypergraphNeuralODE with a UniGCN drift
  2. Define a target-matching loss (MSE)
  3. Train with optax.adam + equinox
  4. Visualize the learned trajectory

Requires: pip install hgx[dynamics] optax
"""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import optax
from hgx._dynamics import HypergraphNeuralODE


def main():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    # --- Hypergraph setup ---
    hg = hgx.from_edge_list(
        [(0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 3, 4)],
        num_nodes=5,
        node_features=jax.random.normal(k1, (5, 4)),
    )
    target = jax.random.normal(k2, (5, 4))

    # --- Model: Neural ODE with UniGCN drift ---
    conv = hgx.UniGCNConv(in_dim=4, out_dim=4, key=k3)
    model = HypergraphNeuralODE(conv)

    # --- Optimizer ---
    optim = optax.adam(1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # --- Training step ---
    @eqx.filter_jit
    def step(model, opt_state, hg, target):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            sol = m(hg, t0=0.0, t1=1.0)
            return jnp.mean((sol.ys[-1] - target) ** 2)

        loss, grads = loss_fn(model)
        updates, opt_state_new = optim.update(grads, opt_state)
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    # --- Training loop ---
    losses = []
    for i in range(100):
        model, opt_state, loss = step(model, opt_state, hg, target)
        losses.append(float(loss))
        if i % 20 == 0:
            print(f"Step {i:3d} | Loss: {loss:.6f}")

    print(f"Final   | Loss: {losses[-1]:.6f}")

    # --- Trajectory visualization ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import diffrax
        import matplotlib.pyplot as plt

        ts = jnp.linspace(0.0, 1.0, 50)
        sol = model(hg, t0=0.0, t1=1.0, saveat=diffrax.SaveAt(ts=ts))
        traj = sol.ys  # (50, 5, 4)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(losses)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("MSE Loss")
        ax1.set_title("Training loss")
        ax1.set_yscale("log")

        # Feature trajectories for node 0 (dashed = target)
        for d in range(4):
            ax2.plot(ts, traj[:, 0, d], label=f"dim {d}")
            ax2.axhline(target[0, d], color=f"C{d}", ls="--", alpha=0.5)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Feature value")
        ax2.set_title("Node 0 trajectory (dashed = target)")
        ax2.legend()

        fig.tight_layout()
        fig.savefig("examples/neural_ode_training.png", dpi=150)
        print("Saved plot to examples/neural_ode_training.png")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
