"""Visualization utilities for hypergraphs.

Requires optional dependencies: ``pip install hgx[viz]``
(matplotlib and networkx).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hgx._hypergraph import Hypergraph


if TYPE_CHECKING:
    import matplotlib.axes

    from hgx._conv._unigat import UniGATConv


def _ensure_deps():
    """Lazily import and return (matplotlib.pyplot, networkx)."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib and networkx. "
            "Install with: pip install hgx[viz]"
        ) from exc
    return plt, nx


def _default_edge_colors(num_edges: int):
    """Return a list of distinct colors for hyperedges."""
    import matplotlib

    name = "tab10" if num_edges <= 10 else "tab20"
    cmap = matplotlib.colormaps[name]
    return [cmap(i % cmap.N) for i in range(num_edges)]


def draw_hypergraph(
    hg: Hypergraph,
    ax: matplotlib.axes.Axes | None = None,
    node_color: str | Sequence[Any] | None = None,
    edge_color: Sequence[Any] | None = None,
    node_labels: Sequence[str] | None = None,
    title: str | None = None,
    edge_linewidth: np.ndarray | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Draw a hypergraph using its star-expansion bipartite layout.

    Vertex nodes are drawn as circles, hyperedge nodes as squares.
    Each hyperedge gets a distinct color applied to its square and
    connecting lines.

    Args:
        hg: Hypergraph to visualize.
        ax: Matplotlib axes. Created if *None*.
        node_color: Color(s) for vertex nodes.
        edge_color: Color(s) for hyperedge nodes and their connecting lines.
        node_labels: Labels for vertex nodes.
        title: Plot title.
        edge_linewidth: Per-membership line widths of shape (nnz,) matching
            the star-expansion ordering. If *None*, uniform width is used.
        **kwargs: Forwarded to ``nx.draw_networkx_nodes`` for vertex nodes.

    Returns:
        The matplotlib axes used for drawing.
    """
    plt, nx = _ensure_deps()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))  # pyright: ignore[reportAssignmentType]
    if ax is None:
        raise ValueError("Could not create axes")

    H = np.asarray(hg._masked_incidence())
    n, m = H.shape

    # Build bipartite star expansion graph
    G = nx.Graph()
    v_nodes = [f"v{i}" for i in range(n)]
    e_nodes = [f"e{k}" for k in range(m)]
    G.add_nodes_from(v_nodes, bipartite=0)
    G.add_nodes_from(e_nodes, bipartite=1)

    memberships: list[tuple[str, str]] = []
    for i in range(n):
        for k in range(m):
            if H[i, k] > 0:
                memberships.append((f"v{i}", f"e{k}"))
    G.add_edges_from(memberships)

    # Layout
    if hg.positions is not None and hg.positions.shape[-1] == 2:
        pos_arr = np.asarray(hg.positions)
        pos = {f"v{i}": pos_arr[i] for i in range(n)}
        # Place hyperedge nodes at the centroid of their member vertices
        for k in range(m):
            members = np.where(H[:, k] > 0)[0]
            if len(members) > 0:
                centroid = pos_arr[members].mean(axis=0)
                # Offset slightly so they don't overlap vertex nodes
                pos[f"e{k}"] = centroid + np.array([0.0, 0.15])
            else:
                pos[f"e{k}"] = np.array([0.0, 0.0])
    else:
        pos = nx.spring_layout(G, seed=42)

    # Colors
    v_color = node_color if node_color is not None else "steelblue"
    e_colors = (
        list(edge_color) if edge_color is not None else _default_edge_colors(m)
    )

    # Draw vertex nodes (circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=v_nodes,
        node_color=v_color if isinstance(v_color, str) else list(v_color),
        node_shape="o",
        node_size=kwargs.pop("node_size", 400),
        ax=ax,
        **kwargs,
    )

    # Draw hyperedge nodes (squares)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=e_nodes,
        node_color=e_colors[:m],
        node_shape="s",
        node_size=300,
        ax=ax,
    )

    # Draw edges colored by hyperedge, with optional per-edge line widths
    edge_to_color = {}
    for k in range(m):
        for i in range(n):
            if H[i, k] > 0:
                edge_to_color[(f"v{i}", f"e{k}")] = e_colors[k]

    # Build a per-membership linewidth lookup if provided
    lw_lookup: dict[tuple[str, str], float] | None = None
    if edge_linewidth is not None:
        lw_arr = np.asarray(edge_linewidth)
        v_idx, e_idx = np.nonzero(H)
        lw_lookup = {}
        for idx in range(len(v_idx)):
            lw_lookup[(f"v{v_idx[idx]}", f"e{e_idx[idx]}")] = float(lw_arr[idx])

    for edge in memberships:
        color = edge_to_color.get(edge, "gray")
        lw = lw_lookup[edge] if lw_lookup is not None else 1.5
        nx.draw_networkx_edges(
            G, pos, edgelist=[edge], edge_color=[color], width=lw, ax=ax
        )

    # Labels
    if node_labels is not None:
        v_labels = {f"v{i}": str(node_labels[i]) for i in range(n)}
    else:
        v_labels = {f"v{i}": str(i) for i in range(n)}
    e_labels = {f"e{k}": f"e{k}" for k in range(m)}

    nx.draw_networkx_labels(G, pos, labels={**v_labels, **e_labels}, ax=ax)

    if title is not None:
        ax.set_title(title)
    ax.axis("off")

    return ax


def draw_incidence(
    hg: Hypergraph,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Visualize the incidence matrix as a heatmap.

    Args:
        hg: Hypergraph whose incidence matrix to display.
        ax: Matplotlib axes. Created if *None*.
        title: Plot title.

    Returns:
        The matplotlib axes used for drawing.
    """
    plt, _nx = _ensure_deps()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))  # pyright: ignore[reportAssignmentType]
    if ax is None:
        raise ValueError("Could not create axes")

    H = np.asarray(hg._masked_incidence())
    n, m = H.shape

    im = ax.imshow(H, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Hyperedges")
    ax.set_ylabel("Vertices")
    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Incidence Matrix")

    return ax


def draw_attention(
    hg: Hypergraph,
    conv: UniGATConv,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Draw a hypergraph with edge thickness proportional to attention weights.

    Runs the UniGATConv forward pass to extract attention coefficients,
    then draws the hypergraph with line widths scaled by attention.

    Args:
        hg: Input hypergraph.
        conv: A UniGATConv layer (used to compute attention).
        ax: Matplotlib axes. Created if *None*.
        title: Plot title.
        **kwargs: Forwarded to :func:`draw_hypergraph`.

    Returns:
        The matplotlib axes used for drawing.
    """
    # Re-derive attention weights (mirrors UniGATConv.__call__)
    H = hg._masked_incidence()
    x = jax.vmap(conv.linear)(hg.node_features)

    out_dim = x.shape[-1]
    a_l = conv.attn[:out_dim]
    a_r = conv.attn[out_dim:]

    if conv.normalize:
        d_e = jnp.sum(H, axis=0, keepdims=True)
        d_e_inv = jnp.where(d_e > 0, 1.0 / d_e, 0.0)
        e = (H * d_e_inv).T @ x
    else:
        e = H.T @ x

    v_score = x @ a_l
    e_score = e @ a_r
    raw_attn = v_score[:, None] + e_score[None, :]
    raw_attn = jnp.where(raw_attn >= 0, raw_attn, conv.negative_slope * raw_attn)

    mask = H > 0
    raw_attn = jnp.where(mask, raw_attn, -1e9)
    attn_weights = jax.nn.softmax(raw_attn, axis=1)
    attn_weights = attn_weights * H

    # Extract per-membership attention as linewidths
    H_np = np.asarray(H)
    attn_np = np.asarray(attn_weights)
    v_idx, e_idx = np.nonzero(H_np)
    attn_vals = attn_np[v_idx, e_idx]

    # Scale to reasonable line widths (0.5 to 5.0)
    if attn_vals.max() > attn_vals.min():
        scaled = 0.5 + 4.5 * (attn_vals - attn_vals.min()) / (
            attn_vals.max() - attn_vals.min()
        )
    else:
        scaled = np.full_like(attn_vals, 2.0)

    return draw_hypergraph(
        hg,
        ax=ax,
        title=title or "Attention Weights",
        edge_linewidth=scaled,
        **kwargs,
    )


def draw_trajectory(
    ts: np.ndarray,
    features: np.ndarray,
    node_indices: Sequence[int] | None = None,
    feature_indices: Sequence[int] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Plot node feature trajectories over time.

    Args:
        ts: Time points of shape ``(T,)``.
        features: Node features of shape ``(T, n, d)``.
        node_indices: Which nodes to plot. Defaults to all.
        feature_indices: Which feature dimensions to plot. Defaults to
            the first dimension only.
        ax: Matplotlib axes. Created if *None*.
        title: Plot title.

    Returns:
        The matplotlib axes used for drawing.
    """
    plt, _nx = _ensure_deps()
    import matplotlib

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))  # pyright: ignore[reportAssignmentType]
    if ax is None:
        raise ValueError("Could not create axes")

    ts_np = np.asarray(ts)
    feat_np = np.asarray(features)
    _T, n, d = feat_np.shape

    nodes = list(node_indices) if node_indices is not None else list(range(n))
    fdims = list(feature_indices) if feature_indices is not None else [0]

    cmap = matplotlib.colormaps["tab10"]
    linestyles = ["-", "--", "-.", ":"]

    for ni, node in enumerate(nodes):
        color = cmap(ni % cmap.N)
        for fi, fdim in enumerate(fdims):
            ls = linestyles[fi % len(linestyles)]
            label = (
                f"node {node}"
                if len(fdims) == 1
                else f"node {node}, dim {fdim}"
            )
            ax.plot(
                ts_np, feat_np[:, node, fdim],
                color=color, linestyle=ls, label=label,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Feature value")
    ax.legend(fontsize=7, ncol=2)

    if title is not None:
        ax.set_title(title)

    return ax


def draw_phase_portrait(
    features: np.ndarray,
    node_indices: Sequence[int] | None = None,
    dims: tuple[int, int] = (0, 1),
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Draw a 2D phase portrait of node feature trajectories.

    Args:
        features: Node features of shape ``(T, n, d)``.
        node_indices: Which nodes to plot. Defaults to all.
        dims: Pair of feature dimensions for the x and y axes.
        ax: Matplotlib axes. Created if *None*.
        title: Plot title.

    Returns:
        The matplotlib axes used for drawing.
    """
    plt, _nx = _ensure_deps()
    import matplotlib

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))  # pyright: ignore[reportAssignmentType]
    if ax is None:
        raise ValueError("Could not create axes")

    feat_np = np.asarray(features)
    _T, n, _d = feat_np.shape
    d0, d1 = dims

    nodes = list(node_indices) if node_indices is not None else list(range(n))
    cmap = matplotlib.colormaps["tab10"]

    for ni, node in enumerate(nodes):
        color = cmap(ni % cmap.N)
        x = feat_np[:, node, d0]
        y = feat_np[:, node, d1]
        ax.plot(x, y, color=color, label=f"node {node}", linewidth=1.2)
        # Start marker
        ax.plot(x[0], y[0], "o", color=color, markersize=6)
        # Direction arrows at midpoint
        mid = len(x) // 2
        if mid > 0 and mid < len(x) - 1:
            ax.annotate(
                "",
                xy=(x[mid + 1], y[mid + 1]),
                xytext=(x[mid], y[mid]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )

    ax.set_xlabel(f"Feature dim {d0}")
    ax.set_ylabel(f"Feature dim {d1}")
    ax.legend(fontsize=7)

    if title is not None:
        ax.set_title(title)

    return ax
