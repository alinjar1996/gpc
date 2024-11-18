from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """A simple multi-layer perceptron."""

    def __init__(self, layer_sizes: Sequence[int], rngs: nnx.Rngs):
        """Initialize the network.

        Args:
            layer_sizes: Sizes of all layers, including input and output.
            rngs: Random number generators for initialization.
        """
        self.num_hidden = len(layer_sizes) - 2

        # TODO: use nnx.scan to scan over layers, reducing compile times
        for i, (input_size, output_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            setattr(
                self, f"l{i}", nnx.Linear(input_size, output_size, rngs=rngs)
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the network."""
        for i in range(self.num_hidden):
            x = getattr(self, f"l{i}")(x)
            x = nnx.swish(x)
        x = getattr(self, f"l{self.num_hidden}")(x)
        return x


class DenoisingMLP(nnx.Module):
    """A simple multi-layer perceptron for action sequence denoising.

    Computes U* = NNet(U, y, t), where U is the noisy action sequence, y is the
    initial observation, and t is the time step in the denoising process.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        hidden_layers: Sequence[int],
        rngs: nnx.Rngs,
    ):
        """Initialize the network.

        Args:
            action_size: Dimension of the actions (u).
            observation_size: Dimension of the observations (y).
            horizon: Number of steps in the action sequence (U = [u0, u1, ...]).
            hidden_layers: Sizes of all hidden layers.
            rngs: Random number generators for initialization.
        """
        self.action_size = action_size
        self.observation_size = observation_size
        self.horizon = horizon

        input_size = horizon * action_size + observation_size + 1
        output_size = horizon * action_size
        self.mlp = MLP(
            [input_size] + list(hidden_layers) + [output_size], rngs=rngs
        )

    def __call__(self, u: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Forward pass through the network."""
        batches = u.shape[:-2]
        u_flat = u.reshape(batches + (self.horizon * self.action_size,))
        x = jnp.concatenate([u_flat, y, t], axis=-1)
        x = self.mlp(x)
        return x.reshape(batches + (self.horizon, self.action_size))
