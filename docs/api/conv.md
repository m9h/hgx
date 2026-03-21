---
category: research
section: methodology
weight: 10
title: "Convolution Layers"
status: draft
slide_summary: "HGX provides six hypergraph convolution layers (UniGCN, UniGAT, UniGIN, THNN, and sparse variants) sharing a unified interface, enabling first-order message passing, attention-weighted aggregation, and tensorized higher-order interactions."
tags: [convolution, message-passing, unigcn, unigat, thnn, hypergraph, methodology]
---

# Convolution Layers

All convolution layers inherit from `AbstractHypergraphConv` and share the same
`__call__(hg: Hypergraph) -> Array` interface.

## AbstractHypergraphConv

::: hgx.AbstractHypergraphConv

---

## First-order layers

### UniGCNConv

::: hgx.UniGCNConv

### UniGATConv

::: hgx.UniGATConv

### UniGINConv

::: hgx.UniGINConv

---

## Higher-order layers

### THNNConv

::: hgx.THNNConv

---

## Sparse variants

Sparse layers use index-based `segment_sum` aggregation over the star expansion
instead of dense matrix multiplication, reducing complexity from O(n*m) to O(nnz).
They are numerically equivalent to their dense counterparts.

### UniGCNSparseConv

::: hgx.UniGCNSparseConv

### THNNSparseConv

::: hgx.THNNSparseConv
