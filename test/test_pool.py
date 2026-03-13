"""Tests for hypergraph pooling and coarsening operators."""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hg(n=8, m=3, d=4, *, key):
    """Create a random hypergraph with n nodes, m edges, d feature dim."""
    k1, k2 = jax.random.split(key)
    H = jax.random.bernoulli(k1, 0.4, shape=(n, m)).astype(jnp.float32)
    # Ensure every edge has at least one node and every node at least one edge
    H = H.at[jnp.arange(min(n, m)), jnp.arange(min(n, m))].set(1.0)
    features = jax.random.normal(k2, (n, d))
    return hgx.from_incidence(H, node_features=features)


# ---------------------------------------------------------------------------
# HypergraphPooling (DiffPool-style)
# ---------------------------------------------------------------------------


class TestHypergraphPooling:
    """Tests for learned soft-assignment pooling."""

    def test_coarsened_dimensions(self, prng_key):
        """Coarsened hypergraph should have num_clusters nodes."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, k2 = jax.random.split(prng_key)
        pool = hgx.HypergraphPooling(in_dim=4, num_clusters=3, key=k1)
        coarsened, S = pool(hg)
        assert coarsened.node_features.shape == (3, 4)
        assert coarsened.incidence.shape == (3, 3)

    def test_assignment_rows_sum_to_one(self, prng_key):
        """Each row of S should sum to 1 (softmax along axis=1)."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.HypergraphPooling(in_dim=4, num_clusters=3, key=k1)
        _, S = pool(hg)
        row_sums = jnp.sum(S, axis=1)
        assert jnp.allclose(row_sums, jnp.ones(8), atol=1e-5)

    def test_link_pred_loss_finite_nonneg(self, prng_key):
        """Link prediction loss should be finite and non-negative."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.HypergraphPooling(in_dim=4, num_clusters=2, key=k1)
        _, S = pool(hg)
        loss = hgx.HypergraphPooling.link_pred_loss(S, hg.incidence)
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_entropy_loss_finite_nonneg(self, prng_key):
        """Entropy loss should be finite and non-negative."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.HypergraphPooling(in_dim=4, num_clusters=2, key=k1)
        _, S = pool(hg)
        loss = hgx.HypergraphPooling.entropy_loss(S)
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_jit_compatible(self, prng_key):
        """HypergraphPooling should work under eqx.filter_jit."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.HypergraphPooling(in_dim=4, num_clusters=2, key=k1)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        coarsened, S = forward(pool, hg)
        assert coarsened.node_features.shape == (2, 4)

    def test_gradient_flow(self, prng_key):
        """Gradients should flow through the soft assignment."""
        hg = _make_hg(n=6, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.HypergraphPooling(in_dim=4, num_clusters=2, key=k1)

        @eqx.filter_grad
        def grad_fn(model):
            coarsened, S = model(hg)
            return jnp.sum(coarsened.node_features)

        grads = grad_fn(pool)
        # The conv linear layer should have non-zero grads
        assert not jnp.allclose(grads.conv.linear.weight, 0.0)


# ---------------------------------------------------------------------------
# TopKPooling
# ---------------------------------------------------------------------------


class TestTopKPooling:
    """Tests for score-based hard pooling."""

    def test_keeps_roughly_half(self, prng_key):
        """With ratio=0.5 on 8 nodes, ~4 nodes should be active."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.TopKPooling(in_dim=4, ratio=0.5, key=k1)
        out = pool(hg)
        active = int(jnp.sum(out.node_mask))
        assert active == 4  # ceil(0.5 * 8) = 4

    def test_node_mask_present(self, prng_key):
        """Output should have a node_mask."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.TopKPooling(in_dim=4, ratio=0.5, key=k1)
        out = pool(hg)
        assert out.node_mask is not None

    def test_edge_mask_present(self, prng_key):
        """Output should have an edge_mask."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.TopKPooling(in_dim=4, ratio=0.5, key=k1)
        out = pool(hg)
        assert out.edge_mask is not None

    def test_shape_preserved(self, prng_key):
        """TopK uses masks so array shapes should not change."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.TopKPooling(in_dim=4, ratio=0.5, key=k1)
        out = pool(hg)
        assert out.node_features.shape == (8, 4)
        assert out.incidence.shape == (8, 3)

    def test_jit_compatible(self, prng_key):
        """TopKPooling should work under eqx.filter_jit."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        pool = hgx.TopKPooling(in_dim=4, ratio=0.5, key=k1)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out = forward(pool, hg)
        assert out.node_mask is not None
        assert int(jnp.sum(out.node_mask)) == 4


# ---------------------------------------------------------------------------
# SpectralPooling
# ---------------------------------------------------------------------------


class TestSpectralPooling:
    """Tests for spectral clustering based pooling."""

    def test_produces_valid_clusters(self, prng_key):
        """Assignments should be in [0, num_clusters) for all nodes."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        pool = hgx.SpectralPooling(num_clusters=3)
        coarsened, assignments = pool(hg)
        assert assignments.shape == (8,)
        assert jnp.all(assignments >= 0)
        assert jnp.all(assignments < 3)

    def test_coarsened_dimensions(self, prng_key):
        """Coarsened hypergraph should have num_clusters nodes."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        pool = hgx.SpectralPooling(num_clusters=3)
        coarsened, _ = pool(hg)
        assert coarsened.node_features.shape == (3, 4)
        assert coarsened.incidence.shape == (3, 3)

    def test_jit_compatible(self, prng_key):
        """SpectralPooling should work under eqx.filter_jit."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        pool = hgx.SpectralPooling(num_clusters=2)

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        coarsened, assignments = forward(pool, hg)
        assert coarsened.node_features.shape == (2, 4)


# ---------------------------------------------------------------------------
# HierarchicalHGNN
# ---------------------------------------------------------------------------


class TestHierarchicalHGNN:
    """Tests for the full hierarchical model."""

    def test_forward_produces_graph_level_output(self, prng_key):
        """Output should be a 1-d vector of shape (out_dim,)."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        model = hgx.HierarchicalHGNN(
            conv_dims=[(4, 8), (8, 8), (8, 4)],
            pool_clusters=[4, 2],
            out_dim=5,
            key=k1,
        )
        out = model(hg)
        assert out.shape == (5,)

    def test_jit_compatible(self, prng_key):
        """HierarchicalHGNN should work under eqx.filter_jit."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        model = hgx.HierarchicalHGNN(
            conv_dims=[(4, 8), (8, 4)],
            pool_clusters=[3],
            out_dim=2,
            key=k1,
        )

        @eqx.filter_jit
        def forward(model, hg):
            return model(hg)

        out = forward(model, hg)
        assert out.shape == (2,)

    def test_gradient_flow(self, prng_key):
        """Gradients should flow end-to-end through the model."""
        hg = _make_hg(n=8, m=3, d=4, key=prng_key)
        k1, _ = jax.random.split(prng_key)
        model = hgx.HierarchicalHGNN(
            conv_dims=[(4, 8), (8, 4)],
            pool_clusters=[3],
            out_dim=2,
            key=k1,
        )

        @eqx.filter_grad
        def grad_fn(model):
            return jnp.sum(model(hg))

        grads = grad_fn(model)
        # Readout linear should have gradients
        assert not jnp.allclose(grads.readout.weight, 0.0)


# ---------------------------------------------------------------------------
# hypergraph_global_pool
# ---------------------------------------------------------------------------


class TestGlobalPool:
    """Tests for global (non-learned) pooling."""

    def test_mean_shape(self, prng_key):
        """Mean pool should produce shape (d,) from (n, d)."""
        hg = _make_hg(n=6, m=2, d=4, key=prng_key)
        out = hgx.hypergraph_global_pool(hg, method="mean")
        assert out.shape == (4,)

    def test_sum_shape(self, prng_key):
        """Sum pool should produce shape (d,)."""
        hg = _make_hg(n=6, m=2, d=4, key=prng_key)
        out = hgx.hypergraph_global_pool(hg, method="sum")
        assert out.shape == (4,)

    def test_max_shape(self, prng_key):
        """Max pool should produce shape (d,)."""
        hg = _make_hg(n=6, m=2, d=4, key=prng_key)
        out = hgx.hypergraph_global_pool(hg, method="max")
        assert out.shape == (4,)

    def test_mean_value(self):
        """Mean over uniform features should return that value."""
        H = jnp.ones((4, 1))
        features = jnp.ones((4, 3)) * 2.0
        hg = hgx.from_incidence(H, node_features=features)
        out = hgx.hypergraph_global_pool(hg, method="mean")
        assert jnp.allclose(out, jnp.full(3, 2.0), atol=1e-6)

    def test_sum_value(self):
        """Sum over uniform features should return n * value."""
        H = jnp.ones((4, 1))
        features = jnp.ones((4, 3)) * 2.0
        hg = hgx.from_incidence(H, node_features=features)
        out = hgx.hypergraph_global_pool(hg, method="sum")
        assert jnp.allclose(out, jnp.full(3, 8.0), atol=1e-6)

    def test_respects_node_mask(self):
        """Masked nodes should be excluded from the aggregation."""
        H = jnp.ones((4, 1))
        features = jnp.array([
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
            [100.0, 0.0],  # masked out
        ])
        mask = jnp.array([True, True, True, False])
        hg = hgx.Hypergraph(
            node_features=features,
            incidence=H,
            node_mask=mask,
        )
        out = hgx.hypergraph_global_pool(hg, method="mean")
        assert jnp.allclose(out, jnp.array([3.0, 0.0]), atol=1e-6)
