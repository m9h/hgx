"""hgx: Hypergraph neural networks in JAX/Equinox.

Topological and geometric deep learning on higher-order domains.
"""

import importlib.metadata

from hgx._conv import (
    AbstractHypergraphConv as AbstractHypergraphConv,
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
from hgx._ndp import (
    CellProgram as CellProgram,
    develop_trajectory as develop_trajectory,
    HypergraphNDP as HypergraphNDP,
)
from hgx._sparse import (
    edge_to_vertex as edge_to_vertex,
    incidence_to_star_expansion as incidence_to_star_expansion,
    vertex_to_edge as vertex_to_edge,
)
from hgx._transforms import (
    clique_expansion as clique_expansion,
    hypergraph_laplacian as hypergraph_laplacian,
)


try:
    from hgx._dynamics import (
        evolve as evolve,
        HypergraphNeuralCDE as HypergraphNeuralCDE,
        HypergraphNeuralODE as HypergraphNeuralODE,
        HypergraphNeuralSDE as HypergraphNeuralSDE,
        trajectory as trajectory,
    )
    from hgx._latent import (
        LatentHypergraphODE as LatentHypergraphODE,
        LatentHypergraphSDE as LatentHypergraphSDE,
    )
except ImportError:
    pass

try:
    from hgx._data import (
        load_cell_lineage as load_cell_lineage,
        load_connectome as load_connectome,
        load_devograph as load_devograph,
        load_synthetic_karate as load_synthetic_karate,
    )
except ImportError:
    pass

try:
    from hgx._viz import (
        DrawConfig as DrawConfig,
        draw_attention as draw_attention,
        draw_hypergraph as draw_hypergraph,
        draw_incidence as draw_incidence,
        draw_phase_portrait as draw_phase_portrait,
        draw_trajectory as draw_trajectory,
    )
except ImportError:
    pass

try:
    from hgx._pgmax import (
        ActiveInferenceStep as ActiveInferenceStep,
        hypergraph_to_factor_graph as hypergraph_to_factor_graph,
        learn_potentials as learn_potentials,
        run_cell_fate_inference as run_cell_fate_inference,
    )
except ImportError:
    pass

__version__ = importlib.metadata.version("hgx")
