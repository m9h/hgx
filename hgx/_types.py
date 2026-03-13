"""Core type definitions for hgx."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


# Array type aliases for clarity
NodeFeatures = Float[Array, "num_nodes node_dim"]
EdgeFeatures = Float[Array, "num_edges edge_dim"]
Incidence = Float[Array, "num_nodes num_edges"]
Positions = Float[Array, "num_nodes spatial_dim"]
Mask = Bool[Array, "num_items"]
Indices = Int[Array, "num_entries"]
