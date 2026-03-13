import time

import hgx
import jax
import jax.numpy as jnp
import pytest


@pytest.mark.slow
def test_full_workflow_integration():
    """
    End-to-end integration test:
    - build a hypergraph
    - run conv
    - run Neural ODE
    - visualize
    - verify no errors and total execution under 30 seconds
    """

    start_time = time.time()

    # 1. Build a hypergraph
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)

    num_nodes = 8
    feat_dim = 16
    node_features = jax.random.normal(k1, (num_nodes, feat_dim))

    edges = [(0, 1, 2), (2, 3, 4), (4, 5, 6), (1, 6, 7)]
    hg = hgx.from_edge_list(
        edges,
        num_nodes=num_nodes,
        node_features=node_features,
    )

    assert hg.num_nodes == num_nodes
    assert hg.num_edges == len(edges)

    # 2. Run a conv layer
    conv = hgx.UniGCNConv(in_dim=feat_dim, out_dim=feat_dim, key=k2)
    out = conv(hg)
    assert out.shape == (num_nodes, feat_dim)
    assert not jnp.any(jnp.isnan(out))

    # 3. Run Neural ODE (requires diffrax)
    # The environment should have diffrax if the extra [dynamics] is installed.
    # Otherwise skip if diffrax is missing.
    try:
        import diffrax  # noqa: F401
        neural_ode = hgx.HypergraphNeuralODE(conv)
        sol = neural_ode(hg, t0=0.0, t1=1.0, dt0=0.1)

        _trajectory = sol.ys
        # sol.ys shape should be (T, num_nodes, feat_dim) or similar.
        # It's usually (num_nodes, feat_dim) if we just save at t1, but diffrax
        # saves whatever we ask it to. Default SaveAt is t1=True.
        # We can also test hgx.evolve:
        hg_evolved = hgx.evolve(neural_ode, hg, t0=0.0, t1=1.0)
        assert hg_evolved.node_features.shape == (num_nodes, feat_dim)
        assert not jnp.any(jnp.isnan(hg_evolved.node_features))

    except ImportError:
        pytest.skip("diffrax not installed, skipping Neural ODE integration")

    # 4. Visualize (with Agg backend)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx  # noqa: F401

        # Test draw_hypergraph
        ax = hgx.draw_hypergraph(hg, title="Integration Test Hypergraph")
        assert ax is not None
        plt.close(ax.figure)

        # Test draw_incidence
        ax_inc = hgx.draw_incidence(hg)
        assert ax_inc is not None
        plt.close(ax_inc.figure)

    except ImportError:
        pytest.skip("viz dependencies missing, skipping visualization")

    end_time = time.time()
    elapsed = end_time - start_time

    # Verify execution time
    assert elapsed < 30.0, f"Integration test took too long: {elapsed:.2f}s"
