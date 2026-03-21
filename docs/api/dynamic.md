---
category: research
section: methodology
weight: 30
title: "Dynamic Topology"
status: draft
slide_summary: "JIT-compatible dynamic topology operations (add/remove nodes and hyperedges) use pre-allocated masked arrays so that array shapes remain fixed, enabling topology changes within JAX-compiled functions."
tags: [dynamic-topology, jit, preallocate, hypergraph, methodology]
---

# Dynamic Topology

JIT-compatible operations for modifying hypergraph topology at runtime.
All operations use pre-allocated masked arrays so that array shapes never change.

## preallocate

::: hgx.preallocate

## add_node

::: hgx.add_node

## add_hyperedge

::: hgx.add_hyperedge

## remove_node

::: hgx.remove_node

## remove_hyperedge

::: hgx.remove_hyperedge
