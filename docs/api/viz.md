---
category: research
section: methodology
weight: 60
title: "Visualization"
status: draft
slide_summary: "Visualization utilities render hypergraphs as bipartite star-expansion layouts, incidence matrix heatmaps, and attention weight diagrams using matplotlib and networkx."
tags: [visualization, matplotlib, networkx, hypergraph-drawing, methodology]
---

# Visualization

Hypergraph visualization utilities. Requires optional dependencies:

```bash
pip install hgx[viz]
```

!!! note
    The visualization functions are only available when `matplotlib` and
    `networkx` are installed. They are imported conditionally in `hgx.__init__`.

## draw_hypergraph

::: hgx.draw_hypergraph

## draw_incidence

::: hgx.draw_incidence

## draw_attention

::: hgx.draw_attention
