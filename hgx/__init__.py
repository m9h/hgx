"""hgx: Hypergraph neural networks in JAX/Equinox.

Topological and geometric deep learning on higher-order domains.
"""

import importlib.metadata

from hgx._conv import (
    AbstractHypergraphConv as AbstractHypergraphConv,
    HGNNConv as HGNNConv,
    HGNNSparseConv as HGNNSparseConv,
    LorentzHypergraphConv as LorentzHypergraphConv,
    PoincareHypergraphConv as PoincareHypergraphConv,
    ProductHypergraphConv as ProductHypergraphConv,
    ProductManifold as ProductManifold,
    ProductManifoldConv as ProductManifoldConv,
    ProductManifoldMLP as ProductManifoldMLP,
    ProductSpaceConv as ProductSpaceConv,
    ProductSpaceEmbedding as ProductSpaceEmbedding,
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
from hgx._info_geometry import (
    fisher_rao_distance as fisher_rao_distance,
    fisher_rao_metric as fisher_rao_metric,
    free_energy as free_energy,
    info_belief_update as info_belief_update,
    InfoGeometricDynamics as InfoGeometricDynamics,
    js_divergence as js_divergence,
    kl_divergence as kl_divergence,
    natural_gradient as natural_gradient,
    natural_gradient_descent as natural_gradient_descent,
    symmetrized_kl as symmetrized_kl,
    wasserstein_on_simplex as wasserstein_on_simplex,
)
from hgx._model import (
    HGNNStack as HGNNStack,
)
from hgx._ndp import (
    CellProgram as CellProgram,
    develop_trajectory as develop_trajectory,
    HypergraphNDP as HypergraphNDP,
)
from hgx._ot import (
    feature_cost_matrix as feature_cost_matrix,
    gromov_wasserstein as gromov_wasserstein,
    hypergraph_gromov_wasserstein as hypergraph_gromov_wasserstein,
    hypergraph_wasserstein as hypergraph_wasserstein,
    ot_hyperedge_aggregation as ot_hyperedge_aggregation,
    ot_hypergraph_alignment as ot_hypergraph_alignment,
    OTConv as OTConv,
    OTLayer as OTLayer,
    sinkhorn as sinkhorn,
    structural_cost_matrix as structural_cost_matrix,
    unbalanced_sinkhorn as unbalanced_sinkhorn,
    wasserstein_barycenter as wasserstein_barycenter,
    wasserstein_distance as wasserstein_distance,
)
from hgx._perturbation import (
    in_silico_knockout as in_silico_knockout,
    perturbation_screen as perturbation_screen,
    PerturbationEncoder as PerturbationEncoder,
    PerturbationPredictor as PerturbationPredictor,
    train_perturbation_predictor as train_perturbation_predictor,
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
from hgx._temporal import (
    align_topologies as align_topologies,
    fit_neural_ode as fit_neural_ode,
    from_snapshots as from_snapshots,
    interpolate as interpolate,
    sliding_window as sliding_window,
    temporal_smoothness_loss as temporal_smoothness_loss,
    TemporalHypergraph as TemporalHypergraph,
)
from hgx._topology import (
    hodge_laplacians as hodge_laplacians,
    TopologicalLayer as TopologicalLayer,
)
from hgx._transforms import (
    clique_expansion as clique_expansion,
    hypergraph_laplacian as hypergraph_laplacian,
)
from hgx._wavelets import (
    cheeger_constant_bound as cheeger_constant_bound,
    hypergraph_scattering as hypergraph_scattering,
    hypergraph_wavelet_transform as hypergraph_wavelet_transform,
    HypergraphWaveletLayer as HypergraphWaveletLayer,
    spectral_features as spectral_features,
)

# --- Optional: persistence (requires giotto-tda or ripser) ---
try:
    from hgx._topology import (
        compute_persistence as compute_persistence,
        persistence_features as persistence_features,
        persistence_image as persistence_image,
        persistence_landscape as persistence_landscape,
    )
except ImportError:
    pass

# --- Optional: dynamics (requires diffrax) ---
try:
    from hgx._dynamics import (
        evolve as evolve,
        HypergraphNeuralCDE as HypergraphNeuralCDE,
        HypergraphNeuralODE as HypergraphNeuralODE,
        HypergraphNeuralSDE as HypergraphNeuralSDE,
        trajectory as trajectory,
    )
    from hgx._geometric_dynamics import (
        AbstractManifold as AbstractManifold,
        EuclideanManifold as EuclideanManifold,
        PoincareBall as PoincareBall,
        riemannian_trajectory as riemannian_trajectory,
        RiemannianHypergraphODE as RiemannianHypergraphODE,
    )
    from hgx._info_geometry import (
        FisherRaoDrift as FisherRaoDrift,
        FreeEnergyDrift as FreeEnergyDrift,
        InfoGeometricODE as InfoGeometricODE,
    )
    from hgx._latent import (
        LatentHypergraphODE as LatentHypergraphODE,
        LatentHypergraphSDE as LatentHypergraphSDE,
    )
except ImportError:
    pass

# --- Optional: SE(3) equivariance (requires e3nn-jax) ---
try:
    from hgx._conv._se3 import SE3HypergraphConv as SE3HypergraphConv
except ImportError:
    pass

# --- Optional: GRN loaders ---
try:
    from hgx._grn import (
        grn_to_temporal_hypergraphs as grn_to_temporal_hypergraphs,
        load_grn_from_anndata as load_grn_from_anndata,
        load_grn_from_csv as load_grn_from_csv,
        load_grn_from_edge_list as load_grn_from_edge_list,
    )
except ImportError:
    pass

# --- Optional: PGMax bridge (requires pgmax) ---
try:
    from hgx._pgmax import (
        ActiveInferenceStep as ActiveInferenceStep,
        hypergraph_to_factor_graph as hypergraph_to_factor_graph,
        learn_potentials as learn_potentials,
        run_cell_fate_inference as run_cell_fate_inference,
    )
except ImportError:
    pass

# --- Optional: visualization (requires matplotlib, networkx) ---
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
