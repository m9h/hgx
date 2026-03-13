"""SE(3)-equivariant hypergraph convolution layer.

Performs V->E->V message passing using irreducible representations
and tensor products from e3nn-jax, ensuring equivariance under 3D
rotations and translations.  Geometric information enters via
relative position vectors between nodes (from ``hg.positions``),
which are projected onto spherical harmonics and combined with
node features through tensor products.

Requires the ``geometry`` extra: ``pip install hgx[geometry]``
"""

from __future__ import annotations

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._hypergraph import Hypergraph


class SE3HypergraphConv(AbstractHypergraphConv):
    """SE(3)-equivariant hypergraph convolution.

    Uses e3nn-jax irreducible representations and tensor products to
    build rotationally equivariant messages.  The forward pass:

    1. **V->E (vertex to hyperedge):**  For each hyperedge, compute
       relative position vectors from member nodes to the hyperedge
       centroid, project onto spherical harmonics, tensor-product with
       linearly projected scalar node features, and aggregate.

    2. **E->V (hyperedge to vertex):**  Aggregate edge messages back
       to each incident vertex via a learned equivariant linear map.

    Input node features are assumed scalar (``in_dim x 0e``).
    Geometric (vector/tensor) information is injected via the tensor
    product of scalar node features with spherical harmonics of
    relative position vectors.  The ``sh_lmax`` parameter controls
    which angular orders appear: l=1 gives vectors, l=2 adds
    quadrupoles, etc.

    Architecture::

        scalars ──linear_src──> Kx0e ──⊗ Y_l(r̂)──> TP irreps
                                                       │
                            linear_msg ────────────────┘
                                │
                          aggregate V->E->V
                                │
                            linear_out ──> out_irreps
    """

    linear_src: e3nn.equinox.Linear
    linear_msg: e3nn.equinox.Linear
    linear_out: e3nn.equinox.Linear
    sh_lmax: int = eqx.field(static=True)
    n_scalar_hidden: int = eqx.field(static=True)
    irreps_msg: e3nn.Irreps = eqx.field(static=True)
    irreps_out: e3nn.Irreps = eqx.field(static=True)
    irreps_sh: e3nn.Irreps = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_irreps: str = "0e + 1o",
        hidden_irreps: str = "0e + 1o",
        sh_lmax: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize SE3HypergraphConv.

        Args:
            in_dim: Input feature dimension (scalar features, ``in_dim x 0e``).
            out_irreps: Output irrep specification, e.g. ``"0e + 1o"``
                or ``"2x0e + 1o + 2e"``.
            hidden_irreps: Controls the number of scalar channels fed
                into the tensor product.  Only the ``0e`` multiplicity
                is used (higher-order terms in *hidden_irreps* set the
                scalar channel count).
            sh_lmax: Maximum spherical harmonic degree for encoding
                relative positions.  1 = vectors, 2 = adds quadrupoles, etc.
                The tensor product of scalars with SH up to *sh_lmax*
                produces all irrep orders from 0 to *sh_lmax*.
            key: PRNG key for weight initialization.
        """
        k1, k2, k3 = jax.random.split(key, 3)

        irreps_in = e3nn.Irreps(f"{in_dim}x0e")
        self.irreps_out = e3nn.Irreps(out_irreps)
        self.sh_lmax = sh_lmax
        self.irreps_sh = e3nn.Irreps.spherical_harmonics(sh_lmax)

        # Count scalar multiplicity in hidden irreps for the source projection.
        # Input is scalar, so the equivariant linear_src can only produce scalars.
        hidden = e3nn.Irreps(hidden_irreps)
        self.n_scalar_hidden = max(hidden.count("0e"), 1)
        irreps_scalar_hidden = e3nn.Irreps(f"{self.n_scalar_hidden}x0e")

        # TP of K scalars with SH(0..lmax) gives the message irreps.
        # E.g. 2x0e ⊗ (0e+1o+2e) = 2x0e + 2x1o + 2x2e
        tp_irreps = e3nn.tensor_product(irreps_scalar_hidden, self.irreps_sh)

        # linear_msg maps TP irreps into a working representation that
        # has sufficient angular orders to produce the desired output.
        # We use the TP irreps themselves (which cover orders 0..sh_lmax).
        self.irreps_msg = tp_irreps

        self.linear_src = e3nn.equinox.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_scalar_hidden,
            key=k1,
        )
        self.linear_msg = e3nn.equinox.Linear(
            irreps_in=tp_irreps,
            irreps_out=self.irreps_msg,
            key=k2,
        )
        self.linear_out = e3nn.equinox.Linear(
            irreps_in=self.irreps_msg,
            irreps_out=self.irreps_out,
            key=k3,
        )

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply SE(3)-equivariant hypergraph convolution.

        Args:
            hg: Input hypergraph.  Must have ``positions`` set (shape
                ``(n, 3)``) and ``geometry == "euclidean"``.

        Returns:
            Updated node features as a plain JAX array of shape
            ``(n, irreps_out.dim)``.
        """
        H = hg._masked_incidence()  # (n, m)
        n, m = H.shape
        pos = hg.positions  # (n, 3)

        # --- Scalar projection of input features ---
        x_in = e3nn.IrrepsArray(
            f"{hg.node_features.shape[-1]}x0e", hg.node_features
        )
        x_scalar = jax.vmap(self.linear_src)(x_in)  # (n, n_scalar_hidden)

        # --- V -> E: vertex to hyperedge message passing ---

        # Hyperedge centroids (mean position of member nodes)
        d_e = jnp.sum(H, axis=0, keepdims=True)  # (1, m)
        d_e_safe = jnp.where(d_e > 0, d_e, 1.0)
        centroids = (H / d_e_safe).T @ pos  # (m, 3)

        # Relative vectors: pos[i] - centroid[k] for all (node, edge) pairs
        rel_vecs = pos[:, None, :] - centroids[None, :, :]  # (n, m, 3)

        # Spherical harmonics of relative vectors
        rel_flat = e3nn.IrrepsArray("1o", rel_vecs.reshape(n * m, 3))
        sh_flat = e3nn.spherical_harmonics(
            list(range(self.sh_lmax + 1)),
            rel_flat,
            normalize=True,
        )  # (n*m, sh_dim)

        # Expand scalar features: repeat each node's features m times
        x_rep = e3nn.IrrepsArray(
            x_scalar.irreps,
            jnp.repeat(x_scalar.array, m, axis=0),
        )  # (n*m, n_scalar_hidden)

        # Tensor product injects geometry into scalar features
        tp = e3nn.tensor_product(x_rep, sh_flat)  # (n*m, tp_dim)

        # Learned equivariant projection of TP messages
        msg_flat = jax.vmap(self.linear_msg)(tp)  # (n*m, msg_dim)

        # Reshape to (n, m, msg_dim) and mask by incidence
        msg = msg_flat.array.reshape(n, m, -1)
        msg_masked = msg * H[:, :, None]

        # Mean aggregation over member nodes per hyperedge
        edge_msgs = jnp.sum(msg_masked, axis=0) / jnp.maximum(d_e.T, 1.0)

        # --- E -> V: hyperedge to vertex aggregation ---
        d_v = jnp.sum(H, axis=1, keepdims=True)  # (n, 1)
        d_v_safe = jnp.where(d_v > 0, d_v, 1.0)
        agg = (H @ edge_msgs) / d_v_safe  # (n, msg_dim)

        # Final equivariant linear projection
        agg_irreps = e3nn.IrrepsArray(self.irreps_msg, agg)
        out = jax.vmap(self.linear_out)(agg_irreps)

        result = out.array

        if hg.node_mask is not None:
            result = result * hg.node_mask[:, None]

        return result
