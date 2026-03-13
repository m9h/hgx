"""Tests for visualization utilities (Agg backend, no display)."""

import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import hgx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from hgx._viz import (
    draw_attention,
    draw_hypergraph,
    draw_incidence,
    draw_phase_portrait,
    draw_trajectory,
)


@pytest.fixture
def six_node_hg():
    """6 nodes, 4 hyperedges of varying size."""
    return hgx.from_edge_list(
        [(0, 1, 2), (2, 3), (3, 4, 5), (0, 1, 3, 4, 5)],
        num_nodes=6,
        node_features=jnp.eye(6),
    )


class TestDrawHypergraph:
    def test_returns_axes(self, six_node_hg):
        ax = draw_hypergraph(six_node_hg)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_title(self, six_node_hg):
        ax = draw_hypergraph(six_node_hg, title="test")
        assert ax.get_title() == "test"
        plt.close("all")

    def test_custom_node_labels(self, six_node_hg):
        ax = draw_hypergraph(six_node_hg, node_labels=list("ABCDEF"))
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_positions(self):
        """Use provided 2D positions for vertex layout."""
        pos = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        hg = hgx.from_edge_list(
            [(0, 1, 2)],
            num_nodes=3,
            node_features=jnp.ones((3, 1)),
        )
        hg = hgx.Hypergraph(
            node_features=hg.node_features,
            incidence=hg.incidence,
            positions=pos,
        )
        ax = draw_hypergraph(hg)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_tiny_fixture(self, tiny_hypergraph):
        ax = draw_hypergraph(tiny_hypergraph)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_existing_axes(self, six_node_hg):
        fig, ax = plt.subplots()
        returned = draw_hypergraph(six_node_hg, ax=ax)
        assert returned is ax
        plt.close("all")


class TestDrawIncidence:
    def test_returns_axes(self, six_node_hg):
        ax = draw_incidence(six_node_hg)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_default_title(self, six_node_hg):
        ax = draw_incidence(six_node_hg)
        assert ax.get_title() == "Incidence Matrix"
        plt.close("all")

    def test_custom_title(self, six_node_hg):
        ax = draw_incidence(six_node_hg, title="Custom")
        assert ax.get_title() == "Custom"
        plt.close("all")

    def test_axis_labels(self, six_node_hg):
        ax = draw_incidence(six_node_hg)
        assert ax.get_xlabel() == "Hyperedges"
        assert ax.get_ylabel() == "Vertices"
        plt.close("all")


class TestDrawAttention:
    def test_returns_axes(self, tiny_hypergraph):
        key = jax.random.PRNGKey(0)
        conv = hgx.UniGATConv(in_dim=2, out_dim=4, key=key)
        ax = draw_attention(tiny_hypergraph, conv)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_custom_title(self, tiny_hypergraph):
        key = jax.random.PRNGKey(0)
        conv = hgx.UniGATConv(in_dim=2, out_dim=4, key=key)
        ax = draw_attention(tiny_hypergraph, conv, title="Attn")
        assert ax.get_title() == "Attn"
        plt.close("all")


class TestDrawTrajectory:
    def test_returns_axes(self):
        ts = jnp.linspace(0.0, 1.0, 20)
        features = jnp.ones((20, 4, 3))
        ax = draw_trajectory(ts, features)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_node_and_feature_indices(self):
        ts = jnp.linspace(0.0, 1.0, 10)
        features = jax.random.normal(jax.random.PRNGKey(0), (10, 6, 4))
        ax = draw_trajectory(
            ts, features, node_indices=[0, 2], feature_indices=[1, 3]
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_title(self):
        ts = jnp.linspace(0.0, 1.0, 5)
        features = jnp.zeros((5, 2, 2))
        ax = draw_trajectory(ts, features, title="Traj")
        assert ax.get_title() == "Traj"
        plt.close("all")


class TestDrawPhasePortrait:
    def test_returns_axes(self):
        features = jax.random.normal(jax.random.PRNGKey(0), (20, 4, 3))
        ax = draw_phase_portrait(features)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_custom_dims(self):
        features = jax.random.normal(jax.random.PRNGKey(0), (20, 4, 5))
        ax = draw_phase_portrait(features, dims=(1, 3), node_indices=[0, 2])
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_title(self):
        features = jnp.zeros((10, 2, 2))
        ax = draw_phase_portrait(features, title="Phase")
        assert ax.get_title() == "Phase"
        plt.close("all")
