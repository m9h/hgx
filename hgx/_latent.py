"""Latent hypergraph dynamics (encode-integrate-decode).

Implements the Latent ODE / Latent SDE pattern (Rubanova et al. 2019,
Li et al. 2020) adapted for hypergraphs: encode node features into a
latent space, evolve the latent state with a Neural ODE or SDE whose
vector field is a hypergraph convolution, then decode back.

Requires the ``dynamics`` extra: ``pip install hgx[dynamics]``
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from hgx._conv._base import AbstractHypergraphConv
from hgx._dynamics import HypergraphNeuralODE, HypergraphNeuralSDE
from hgx._hypergraph import Hypergraph


try:
    import diffrax
except ImportError as e:
    raise ImportError(
        "hgx latent dynamics requires diffrax. Install with: pip install hgx[dynamics]"
    ) from e


class LatentHypergraphODE(eqx.Module):
    """Latent ODE on hypergraphs: encode -> integrate -> decode.

    Encodes observed node features to a latent space, evolves them with
    a ``HypergraphNeuralODE``, and decodes back to observation space.

    Attributes:
        encoder: MLP mapping (obs_dim,) -> (latent_dim,) per node.
        dynamics: HypergraphNeuralODE operating in latent space.
        decoder: MLP mapping (latent_dim,) -> (obs_dim,) per node.
    """

    encoder: eqx.nn.MLP
    dynamics: HypergraphNeuralODE
    decoder: eqx.nn.MLP

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        conv_cls: type[AbstractHypergraphConv],
        encoder_depth: int = 1,
        decoder_depth: int = 1,
        activation: Callable = jax.nn.tanh,
        solver: diffrax.AbstractSolver | None = None,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
        conv_kwargs: dict | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize LatentHypergraphODE.

        Args:
            obs_dim: Observed feature dimension.
            latent_dim: Latent space dimension.
            conv_cls: Convolution class for the latent dynamics
                (instantiated as ``conv_cls(latent_dim, latent_dim, ...)``.
            encoder_depth: Number of hidden layers in the encoder MLP.
            decoder_depth: Number of hidden layers in the decoder MLP.
            activation: Activation forwarded to HypergraphNeuralODE.
            solver: Diffrax solver forwarded to HypergraphNeuralODE.
            stepsize_controller: Step-size controller forwarded to
                HypergraphNeuralODE.
            conv_kwargs: Extra kwargs passed to the conv constructor.
            key: PRNG key for initialization.
        """
        k_enc, k_conv, k_dec = jax.random.split(key, 3)

        if conv_kwargs is None:
            conv_kwargs = {}

        self.encoder = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=latent_dim,
            width_size=latent_dim,
            depth=encoder_depth,
            activation=jax.nn.relu,  # pyright: ignore[reportArgumentType]
            key=k_enc,
        )

        conv = conv_cls(latent_dim, latent_dim, key=k_conv, **conv_kwargs)  # pyright: ignore[reportCallIssue]
        self.dynamics = HypergraphNeuralODE(
            conv=conv,
            activation=activation,
            solver=solver,
            stepsize_controller=stepsize_controller,
        )

        self.decoder = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=obs_dim,
            width_size=latent_dim,
            depth=decoder_depth,
            activation=jax.nn.relu,  # pyright: ignore[reportArgumentType]
            key=k_dec,
        )

    def __call__(
        self,
        hg: Hypergraph,
        t0: float = 0.0,
        t1: float = 1.0,
        dt0: float | None = None,
        saveat: diffrax.SaveAt | None = None,
    ) -> Float[Array, "n obs_dim"] | diffrax.Solution:
        """Encode, integrate, decode.

        Args:
            hg: Input hypergraph with node features of shape (n, obs_dim).
            t0: Start time.
            t1: End time.
            dt0: Initial step size (forwarded to dynamics).
            saveat: If provided, returns the raw ``diffrax.Solution``
                with decoded snapshots. If None, returns decoded features
                at t1 as an array of shape (n, obs_dim).

        Returns:
            Decoded node features at t1, or a Solution with decoded
            snapshots if ``saveat`` is given.
        """
        # Encode
        z0 = jax.vmap(self.encoder)(hg.node_features)

        # Build latent hypergraph
        latent_hg = Hypergraph(
            node_features=z0,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry=hg.geometry,
        )

        # Integrate
        import typing
        sol = self.dynamics(latent_hg, t0=t0, t1=t1, dt0=dt0, saveat=saveat)

        if saveat is not None:
            return sol

        # Decode final state
        ys = typing.cast(Array, sol.ys)
        z1 = ys[-1]
        return jax.vmap(self.decoder)(z1)


class LatentHypergraphSDE(eqx.Module):
    """Latent SDE on hypergraphs: encode -> stochastic integrate -> decode.

    Same encode-integrate-decode pattern as ``LatentHypergraphODE`` but
    uses ``HypergraphNeuralSDE`` internally, adding learned diagonal
    noise to the latent dynamics.

    Attributes:
        encoder: MLP mapping (obs_dim,) -> (latent_dim,) per node.
        dynamics: HypergraphNeuralSDE operating in latent space.
        decoder: MLP mapping (latent_dim,) -> (obs_dim,) per node.
        num_nodes: Number of nodes (static, for reshape).
        latent_dim: Latent feature dimension (static, for reshape).
    """

    encoder: eqx.nn.MLP
    dynamics: HypergraphNeuralSDE
    decoder: eqx.nn.MLP
    num_nodes: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        num_nodes: int,
        conv_cls: type[AbstractHypergraphConv],
        encoder_depth: int = 1,
        decoder_depth: int = 1,
        activation: Callable = jax.nn.tanh,
        sigma_init: float = 0.1,
        solver: diffrax.AbstractSolver | None = None,
        dt: float = 0.01,
        conv_kwargs: dict | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize LatentHypergraphSDE.

        Args:
            obs_dim: Observed feature dimension.
            latent_dim: Latent space dimension.
            num_nodes: Number of nodes in the hypergraph.
            conv_cls: Convolution class for the latent dynamics.
            encoder_depth: Number of hidden layers in the encoder MLP.
            decoder_depth: Number of hidden layers in the decoder MLP.
            activation: Activation for drift (default: tanh).
            sigma_init: Initial diffusion scale.
            solver: Diffrax SDE solver (default: Euler).
            dt: Fixed step size for SDE integration.
            conv_kwargs: Extra kwargs passed to the conv constructor.
            key: PRNG key for initialization.
        """
        k_enc, k_conv, k_sde, k_dec = jax.random.split(key, 4)

        if conv_kwargs is None:
            conv_kwargs = {}

        self.encoder = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=latent_dim,
            width_size=latent_dim,
            depth=encoder_depth,
            activation=jax.nn.relu,  # pyright: ignore[reportArgumentType]
            key=k_enc,
        )

        conv = conv_cls(latent_dim, latent_dim, key=k_conv, **conv_kwargs)  # pyright: ignore[reportCallIssue]
        self.dynamics = HypergraphNeuralSDE(
            conv=conv,
            num_nodes=num_nodes,
            node_dim=latent_dim,
            activation=activation,
            sigma_init=sigma_init,
            solver=solver,
            dt=dt,
            key=k_sde,
        )

        self.decoder = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=obs_dim,
            width_size=latent_dim,
            depth=decoder_depth,
            activation=jax.nn.relu,  # pyright: ignore[reportArgumentType]
            key=k_dec,
        )
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim

    def __call__(
        self,
        hg: Hypergraph,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        key: PRNGKeyArray,
        saveat: diffrax.SaveAt | None = None,
    ) -> Float[Array, "n obs_dim"] | diffrax.Solution:
        """Encode, stochastically integrate, decode.

        Args:
            hg: Input hypergraph with node features of shape (n, obs_dim).
            t0: Start time.
            t1: End time.
            key: PRNG key for Brownian motion sampling.
            saveat: If provided, returns the raw ``diffrax.Solution``.
                If None, returns decoded features at t1.

        Returns:
            Decoded node features at t1, or a Solution if ``saveat``
            is given.
        """
        # Encode
        z0 = jax.vmap(self.encoder)(hg.node_features)

        # Build latent hypergraph
        latent_hg = Hypergraph(
            node_features=z0,
            incidence=hg.incidence,
            edge_features=hg.edge_features,
            positions=hg.positions,
            node_mask=hg.node_mask,
            edge_mask=hg.edge_mask,
            geometry=hg.geometry,
        )

        # Integrate
        import typing
        sol = self.dynamics(latent_hg, t0=t0, t1=t1, key=key, saveat=saveat)

        if saveat is not None:
            return sol

        # Decode final state (SDE state is flattened)
        ys = typing.cast(Array, sol.ys)
        z1 = ys[-1].reshape(self.num_nodes, self.latent_dim)
        return jax.vmap(self.decoder)(z1)
