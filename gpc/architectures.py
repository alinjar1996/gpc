from typing import Sequence

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

        # TODO: use nnx.scan and nnx.vmap to reduce compile times
        for i, (input_size, output_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
        ):
            setattr(
                self, f"l{i}", nnx.Linear(input_size, output_size, rngs=rngs)
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.num_hidden):
            x = getattr(self, f"l{i}")(x)
            x = nnx.swish(x)
        x = getattr(self, f"l{self.num_hidden}")(x)
        return x


# def print_module_summary(module: nn.Module, input_shape: Sequence[int]) -> None:
#     """Print a readable summary of a flax neural network module.

#     Args:
#         module: The flax module to summarize.
#         input_shape: The shape of the input to the module.
#     """
#     # Create a dummy input
#     rng = jax.random.key(0)
#     dummy_input = jnp.ones(input_shape)
#     print(module.tabulate(rng, dummy_input, depth=1))


# class MLP(nn.Module):
#     """A simple pickle-able multi-layer perceptron.

#     Args:
#         layer_sizes: Sizes of all hidden layers and the output layer.
#         activate_final: Whether to apply an activation function to the output.
#         bias: Whether to use a bias in the linear layers.
#     """

#     layer_sizes: Sequence[int]
#     activate_final: bool = False
#     bias: bool = True

#     @nn.compact
#     def __call__(self, x: jax.Array) -> jax.Array:
#         """Forward pass through the network."""
#         # TODO(vincekurtz): consider using jax control flow here. Note that
#         # standard jax control flows (e.g. jax.lax.scan) do not play nicely with
#         # flax, see for example https://github.com/google/flax/discussions/1283.
#         for i, layer_size in enumerate(self.layer_sizes):
#             x = nn.Dense(
#                 layer_size,
#                 use_bias=self.bias,
#                 kernel_init=nn.initializers.lecun_uniform(),
#                 name=f"dense_{i}",
#             )(x)
#             if i != len(self.layer_sizes) - 1:
#                 x = nn.swish(x)
#             if i == len(self.layer_sizes) - 1 and self.activate_final:
#                 x = nn.tanh(x)
#         return x


# class DenoisingMLP(nn.Module):
#     """A pickle-able module for performing action sequence denoising.

#     This computes U <-- NNet(U, y, t), where U is the action sequence, y is the
#     initial observation, and t is the time step in the denoising process.

#     Args:
#         hidden_layers: Sizes of all hidden layers.

#     """

#     # TODO: consider storing num_steps, action_dim as attributes on construction
#     hidden_layers: Sequence[int]

#     @nn.compact
#     def __call__(self, u: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
#         """Forward pass through the network."""
#         batches = u.shape[:-2]
#         num_steps = u.shape[-2]
#         action_dim = u.shape[-1]
#         u_flat = u.reshape(batches + (num_steps * action_dim,))
#         x = jnp.concatenate([u_flat, y, t], axis=-1)
#         x = MLP(self.hidden_layers + (num_steps * action_dim,))(x)
#         return x.reshape((batches) + (num_steps, action_dim))


# class ActionSequenceMLP(nn.Module):
#     """A pickle-able module for generating a sequence of actions.

#     Generates an action sequence U = NNet(y), where y is the observation. The
#     action sequence has shape (num_steps, action_dim).

#     Args:
#         hidden_layers: Sizes of all hidden layers.
#         num_steps: Number of steps in the action sequence.
#         action_dim: Dimension of the action space.
#     """

#     hidden_layers: Sequence[int]
#     num_steps: int
#     action_dim: int

#     @nn.compact
#     def __call__(self, y: jax.Array) -> jax.Array:
#         """Forward pass through the network."""
#         batches = y.shape[:-1]
#         x = MLP(self.hidden_layers + (self.num_steps * self.action_dim,))(y)
#         return x.reshape(batches + (self.num_steps, self.action_dim))
