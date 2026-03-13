# Continuous-Time Dynamics

Neural ODE and Neural SDE modules that evolve node features in continuous time,
using hypergraph convolution layers as the learned vector field.

Requires the `dynamics` extra:

```bash
pip install hgx[dynamics]
```

!!! note
    These classes are only available when [Diffrax](https://docs.kidger.site/diffrax/)
    is installed. They are imported conditionally in `hgx.__init__`.

## HypergraphNeuralODE

::: hgx.HypergraphNeuralODE

## HypergraphNeuralSDE

::: hgx.HypergraphNeuralSDE

## evolve

::: hgx.evolve
