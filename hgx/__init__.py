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

from hgx._grn import (
    grn_to_temporal_hypergraphs as grn_to_temporal_hypergraphs,
    load_grn_from_anndata as load_grn_from_anndata,
    load_grn_from_csv as load_grn_from_csv,
    load_grn_from_edge_list as load_grn_from_edge_list,
    load_pando_modules as load_pando_modules,
)
from hgx._temporal import (
    align_topologies as align_topologies,
    from_snapshots as from_snapshots,
    interpolate as interpolate,
    sliding_window as sliding_window,
    TemporalHypergraph as TemporalHypergraph,
    temporal_smoothness_loss as temporal_smoothness_loss,
)

try:
    from hgx._conv._hyperbolic import (
        LorentzHypergraphConv as LorentzHypergraphConv,
        PoincareHypergraphConv as PoincareHypergraphConv,
    )
except ImportError:
    pass

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
    from hgx._temporal import (
        fit_neural_ode as fit_neural_ode,
    )
except ImportError:
    pass

try:
    from hgx._perturbation import (
        in_silico_knockout as in_silico_knockout,
        perturbation_screen as perturbation_screen,
        PerturbationEncoder as PerturbationEncoder,
        PerturbationPredictor as PerturbationPredictor,
        train_perturbation_predictor as train_perturbation_predictor,
    )
except ImportError:
    pass

try:
    from hgx._topology import (
        compute_persistence as compute_persistence,
        hodge_laplacians as hodge_laplacians,
        persistence_features as persistence_features,
        persistence_image as persistence_image,
        persistence_landscape as persistence_landscape,
        TopologicalLayer as TopologicalLayer,
    )
except ImportError:
    pass

try:
    from hgx._ndp import (
        CellProgram as CellProgram,
        develop_trajectory as develop_trajectory,
        HypergraphNDP as HypergraphNDP,
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
        draw_attention as draw_attention,
        draw_hypergraph as draw_hypergraph,
        draw_incidence as draw_incidence,
        draw_phase_portrait as draw_phase_portrait,
        draw_trajectory as draw_trajectory,
    )
except ImportError:
    pass

__version__ = importlib.metadata.version("hgx")
