"""hgx: Hypergraph neural networks in JAX/Equinox.

Topological and geometric deep learning on higher-order domains.
"""

import importlib.metadata

from hgx._conv import (
    AbstractHypergraphConv as AbstractHypergraphConv,
    SheafDiffusion as SheafDiffusion,
    SheafHypergraphConv as SheafHypergraphConv,
    THNNConv as THNNConv,
    THNNSparseConv as THNNSparseConv,
    UniGATConv as UniGATConv,
    UniGCNConv as UniGCNConv,
    UniGCNSparseConv as UniGCNSparseConv,
    UniGINConv as UniGINConv,
)
from hgx._dynamic import (
    add_hyperedge as add_hyperedge,
    add_node as add_node,
    preallocate as preallocate,
    remove_hyperedge as remove_hyperedge,
    remove_node as remove_node,
)
from hgx._hypergraph import (
    from_adjacency as from_adjacency,
    from_edge_list as from_edge_list,
    from_incidence as from_incidence,
    Hypergraph as Hypergraph,
)
from hgx._model import (
    HGNNStack as HGNNStack,
)
from hgx._pool import (
    HierarchicalHGNN as HierarchicalHGNN,
    hypergraph_global_pool as hypergraph_global_pool,
    HypergraphPooling as HypergraphPooling,
    SpectralPooling as SpectralPooling,
    TopKPooling as TopKPooling,
)
from hgx._sparse import (
    edge_to_vertex as edge_to_vertex,
    incidence_to_star_expansion as incidence_to_star_expansion,
    vertex_to_edge as vertex_to_edge,
)
from hgx._sparse_incidence import (
    from_edge_list_sparse as from_edge_list_sparse,
    from_sparse_incidence as from_sparse_incidence,
    SparseHypergraph as SparseHypergraph,
    SparseUniGCNConv as SparseUniGCNConv,
    to_sparse as to_sparse,
)
from hgx._transforms import (
    clique_expansion as clique_expansion,
    hypergraph_laplacian as hypergraph_laplacian,
)

try:
    from hgx._viz import (
        draw_attention as draw_attention,
        draw_hypergraph as draw_hypergraph,
        draw_incidence as draw_incidence,
        draw_phase_portrait as draw_phase_portrait,
        draw_trajectory as draw_trajectory,
    )
except ImportError:
    pass

__version__ = importlib.metadata.version("hgx")
