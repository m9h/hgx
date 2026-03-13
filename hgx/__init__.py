"""hgx: Hypergraph neural networks in JAX/Equinox.

Topological and geometric deep learning on higher-order domains.
"""

import importlib.metadata

from hgx._hypergraph import (
    Hypergraph as Hypergraph,
    from_adjacency as from_adjacency,
    from_edge_list as from_edge_list,
    from_incidence as from_incidence,
)
from hgx._conv import (
    AbstractHypergraphConv as AbstractHypergraphConv,
    THNNConv as THNNConv,
    UniGCNConv as UniGCNConv,
)
from hgx._transforms import (
    clique_expansion as clique_expansion,
    hypergraph_laplacian as hypergraph_laplacian,
)


__version__ = importlib.metadata.version("hgx")
