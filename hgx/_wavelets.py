"""Spectral wavelet transforms on hypergraphs for multi-resolution analysis.

Hypergraph wavelets are defined via the spectral decomposition of the
hypergraph Laplacian L = U Lambda U^T.  A wavelet at scale s centered at
node i is psi_{s,i} = g(s Lambda) delta_i, where g is a wavelet-generating
kernel.  The wavelet transform Wf(s,i) = <psi_{s,i}, f> provides
multi-scale features useful for graph classification and signal analysis.

This module also provides:
- A learnable Chebyshev polynomial spectral filter (ChebNet-style) that
  avoids eigendecomposition at runtime.
- A wavelet scattering transform that stacks absolute wavelet coefficients
  for provably stable, invariant graph-level features.
- Spectral feature extraction and Cheeger constant bounds from the
  Laplacian spectrum.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._hypergraph import Hypergraph
from hgx._transforms import hypergraph_laplacian


# ---------------------------------------------------------------------------
# Wavelet kernels
# ---------------------------------------------------------------------------


def _mexican_hat(x: Float[Array, ...]) -> Float[Array, ...]:
    """Mexican hat wavelet kernel: g(x) = x * exp(-x)."""
    return x * jnp.exp(-x)


def _heat_kernel(x: Float[Array, ...]) -> Float[Array, ...]:
    """Heat diffusion kernel: g(x) = exp(-x)."""
    return jnp.exp(-x)


def _meyer_kernel(x: Float[Array, ...]) -> Float[Array, ...]:
    """Meyer-like band-pass kernel.

    A smooth band-pass filter that is concentrated around x = 1:
        g(x) = sin(pi/2 * nu(x)) * (1/3 <= x <= 4/3)
             + cos(pi/2 * nu(x/2 - 1)) * (4/3 < x <= 8/3)
    where nu(x) = x^4 * (35 - 84x + 70x^2 - 20x^3) is a smooth bump.

    We use a simplified differentiable approximation suitable for JAX:
        g(x) = sin(pi * x) * exp(-x^2 / 2)
    """
    return jnp.sin(jnp.pi * x) * jnp.exp(-x**2 / 2.0)


_KERNELS = {
    "mexican_hat": _mexican_hat,
    "heat": _heat_kernel,
    "meyer": _meyer_kernel,
}


# ---------------------------------------------------------------------------
# Hypergraph wavelet transform
# ---------------------------------------------------------------------------


def hypergraph_wavelet_transform(
    hg: Hypergraph,
    scales: Float[Array, " S"] | list[float],
    kernel: str = "mexican_hat",
) -> Float[Array, "S n d"]:
    """Compute wavelet coefficients at multiple scales.

    The wavelet operator at scale s is:
        W_s = U diag(g(s * lambda)) U^T

    where L = U Lambda U^T is the eigendecomposition of the hypergraph
    Laplacian, and g is the chosen wavelet kernel.

    Args:
        hg: Input hypergraph with node features of shape (n, d).
        scales: Array or list of wavelet scales.
        kernel: Wavelet kernel name -- one of "mexican_hat", "heat", "meyer".

    Returns:
        Wavelet coefficients of shape (num_scales, n, d).
    """
    scales = jnp.asarray(scales, dtype=jnp.float32)
    L = hypergraph_laplacian(hg, normalized=True)

    # Eigendecomposition: L = U @ diag(eigenvalues) @ U^T
    eigenvalues, U = jnp.linalg.eigh(L)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)  # numerical clip

    g = _KERNELS[kernel]
    x = hg.node_features  # (n, d)

    def _wavelet_at_scale(s):
        # g(s * lambda) for each eigenvalue
        filter_response = g(s * eigenvalues)  # (n,)
        # W_s = U @ diag(filter_response) @ U^T
        W_s = U * filter_response[None, :] @ U.T  # (n, n)
        return W_s @ x  # (n, d)

    coefficients = jax.vmap(_wavelet_at_scale)(scales)  # (S, n, d)
    return coefficients


# ---------------------------------------------------------------------------
# Chebyshev polynomial utilities
# ---------------------------------------------------------------------------


def _chebyshev_filter(
    L: Float[Array, "n n"],
    x: Float[Array, "n d"],
    coeffs: Float[Array, " K"],
) -> Float[Array, "n d"]:
    """Apply a Chebyshev polynomial spectral filter to features.

    Computes h(L) @ x where h(lambda) = sum_k a_k T_k(lambda_tilde),
    and lambda_tilde is the rescaled eigenvalue in [-1, 1].

    The rescaling is lambda_tilde = 2 * lambda / lambda_max - 1.
    We estimate lambda_max = 2.0 for the normalized Laplacian (theoretical
    upper bound).

    Args:
        L: Laplacian matrix of shape (n, n).
        x: Node features of shape (n, d).
        coeffs: Chebyshev coefficients of shape (K,).

    Returns:
        Filtered features of shape (n, d).
    """
    K = coeffs.shape[0]
    n = L.shape[0]

    # Rescale Laplacian to [-1, 1]: L_tilde = L - I
    # (for normalized Laplacian with eigenvalues in [0, 2])
    L_tilde = L - jnp.eye(n)

    # T_0(L_tilde) @ x = x
    T_prev = x  # T_0
    out = coeffs[0] * T_prev

    if K == 1:
        return out

    # T_1(L_tilde) @ x = L_tilde @ x
    T_curr = L_tilde @ x
    out = out + coeffs[1] * T_curr

    def _step(carry, k):
        T_prev, T_curr, out = carry
        T_next = 2.0 * (L_tilde @ T_curr) - T_prev
        out = out + coeffs[k] * T_next
        return (T_curr, T_next, out), None

    if K > 2:
        (_, _, out), _ = jax.lax.scan(
            _step,
            (T_prev, T_curr, out),
            jnp.arange(2, K),
        )

    return out


# ---------------------------------------------------------------------------
# Learnable wavelet convolution layer (ChebNet)
# ---------------------------------------------------------------------------


class HypergraphWaveletLayer(eqx.Module):
    """Learnable spectral filter on hypergraphs via Chebyshev polynomials.

    Implements the ChebNet approach adapted for hypergraphs: the spectral
    filter h(lambda) = sum_k a_k T_k(lambda) is parameterized by learnable
    Chebyshev coefficients, avoiding explicit eigendecomposition at runtime.

    The forward pass computes:
        Y = h(L) @ X @ W + b

    where L is the hypergraph Laplacian, X is the input feature matrix,
    h(L) is the polynomial filter, and W, b are a linear projection.

    Attributes:
        chebyshev_coeffs: Learnable Chebyshev polynomial coefficients of
            shape (K,).
        linear: Linear projection from in_dim to out_dim.
        K: Polynomial order (number of Chebyshev terms).
    """

    chebyshev_coeffs: Float[Array, " K"]
    linear: eqx.nn.Linear
    K: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int = 5,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize HypergraphWaveletLayer.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            K: Polynomial order (number of Chebyshev coefficients).
            key: PRNG key for weight initialization.
        """
        k1, k2 = jax.random.split(key)
        self.K = K
        self.chebyshev_coeffs = jax.random.normal(k1, (K,)) * 0.1
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=k2)

    def __call__(self, hg: Hypergraph) -> Float[Array, "n out_dim"]:
        """Apply learnable spectral filter.

        Args:
            hg: Input hypergraph.

        Returns:
            Filtered node features of shape (num_nodes, out_dim).
        """
        L = hypergraph_laplacian(hg, normalized=True)
        x = hg.node_features  # (n, d)

        # Apply Chebyshev polynomial filter
        filtered = _chebyshev_filter(L, x, self.chebyshev_coeffs)  # (n, d)

        # Linear projection
        out = jax.vmap(self.linear)(filtered)  # (n, out_dim)

        # Zero out masked nodes
        if hg.node_mask is not None:
            out = out * hg.node_mask[:, None]

        return out


# ---------------------------------------------------------------------------
# Wavelet scattering transform
# ---------------------------------------------------------------------------


def hypergraph_scattering(
    hg: Hypergraph,
    scales: Float[Array, " S"] | list[float],
    num_layers: int = 2,
) -> Float[Array, " F"]:
    """Wavelet scattering transform for graph-level features.

    Produces a fixed-size feature vector invariant to permutation and
    stable to deformations, suitable for graph classification.

    The scattering transform cascades wavelet transforms with pointwise
    absolute values:
        Layer 0: |W_{s_1} f| for each scale s_1
        Layer 1: |W_{s_2} |W_{s_1} f|| for each pair (s_1, s_2)
        ...

    Each layer's output is averaged over nodes to produce graph-level
    features, which are concatenated into a single vector.

    This is a pure computation with no learnable parameters.

    Args:
        hg: Input hypergraph with node features of shape (n, d).
        scales: Array or list of wavelet scales.
        num_layers: Number of scattering layers (default 2).

    Returns:
        Fixed-size feature vector of shape (F,) where
        F = d + S*d + S^2*d + ... (for num_layers layers).
    """
    scales = jnp.asarray(scales, dtype=jnp.float32)
    L = hypergraph_laplacian(hg, normalized=True)

    # Eigendecomposition (computed once)
    eigenvalues, U = jnp.linalg.eigh(L)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)

    g = _mexican_hat  # Use Mexican hat kernel for scattering
    num_scales = scales.shape[0]

    def _apply_wavelet(signal, s):
        """Apply wavelet at scale s to a signal matrix (n, d')."""
        filter_response = g(s * eigenvalues)  # (n,)
        W_s = U * filter_response[None, :] @ U.T  # (n, n)
        return W_s @ signal  # (n, d')

    features_list = []

    # Layer 0: mean of original signal (zeroth-order scattering coefficient)
    x = hg.node_features  # (n, d)
    features_list.append(jnp.mean(x, axis=0))  # (d,)

    # Propagate signals through scattering layers
    # current_signals is a list of (n, d) arrays to propagate further
    current_signals = [x]

    for layer in range(num_layers):
        next_signals = []
        for signal in current_signals:
            for s_idx in range(num_scales):
                s = scales[s_idx]
                scattered = jnp.abs(_apply_wavelet(signal, s))  # (n, d)
                features_list.append(jnp.mean(scattered, axis=0))  # (d,)
                next_signals.append(scattered)
        current_signals = next_signals

    return jnp.concatenate(features_list, axis=0)


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------


def spectral_features(
    hg: Hypergraph,
    num_eigvals: int = 10,
) -> Float[Array, " F"]:
    """Extract spectral features from the hypergraph Laplacian.

    Computes a fixed-size feature vector containing:
    - The first ``num_eigvals`` eigenvalues of the normalized Laplacian
    - Spectral gap (lambda_2 - lambda_1, where lambda_1 = 0)
    - Algebraic connectivity (lambda_2)
    - Spectral radius (largest eigenvalue)

    Total feature size: num_eigvals + 3.

    Args:
        hg: Input hypergraph.
        num_eigvals: Number of smallest eigenvalues to include.

    Returns:
        Feature vector of shape (num_eigvals + 3,).
    """
    L = hypergraph_laplacian(hg, normalized=True)

    eigenvalues = jnp.linalg.eigvalsh(L)
    eigenvalues = jnp.sort(eigenvalues)

    # Pad or truncate to num_eigvals
    n = eigenvalues.shape[0]
    padded = jnp.zeros(num_eigvals)
    k = jnp.minimum(n, num_eigvals)
    padded = padded.at[:k].set(eigenvalues[:k])

    # Spectral gap = lambda_2 (since lambda_1 ~ 0 for connected graphs)
    lambda_2 = jnp.where(n >= 2, eigenvalues[1], 0.0)
    spectral_gap = lambda_2  # lambda_2 - lambda_1 where lambda_1 ~ 0
    algebraic_connectivity = lambda_2
    spectral_radius = eigenvalues[-1]

    return jnp.concatenate([
        padded,
        jnp.array([spectral_gap, algebraic_connectivity, spectral_radius]),
    ])


# ---------------------------------------------------------------------------
# Cheeger constant bounds
# ---------------------------------------------------------------------------


def cheeger_constant_bound(
    hg: Hypergraph,
) -> tuple[float, float]:
    """Compute bounds on the hypergraph Cheeger constant from the spectrum.

    The Cheeger inequality relates the spectral gap lambda_2 of the
    normalized Laplacian to the Cheeger constant h(H):

        lambda_2 / 2 <= h(H) <= sqrt(2 * lambda_2)

    This gives both a lower and upper bound on the Cheeger constant
    without computing it combinatorially.

    Args:
        hg: Input hypergraph.

    Returns:
        Tuple (lower_bound, upper_bound) on the Cheeger constant.
    """
    L = hypergraph_laplacian(hg, normalized=True)
    eigenvalues = jnp.linalg.eigvalsh(L)
    eigenvalues = jnp.sort(eigenvalues)

    # lambda_2: second smallest eigenvalue (algebraic connectivity)
    lambda_2 = jnp.where(eigenvalues.shape[0] >= 2, eigenvalues[1], 0.0)
    lambda_2 = jnp.maximum(lambda_2, 0.0)  # numerical safety

    # Cheeger inequality: lambda_2 / 2 <= h <= sqrt(2 * lambda_2)
    lower_bound = lambda_2 / 2.0
    upper_bound = jnp.sqrt(2.0 * lambda_2)

    return float(lower_bound), float(upper_bound)
