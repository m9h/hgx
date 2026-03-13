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
