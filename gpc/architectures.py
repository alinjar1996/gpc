from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def print_module_summary(module: nn.Module, input_shape: Sequence[int]) -> None:
    """Print a readable summary of a flax neural network module.

    Args:
        module: The flax module to summarize.
        input_shape: The shape of the input to the module.
    """
    # Create a dummy input
    rng = jax.random.key(0)
    dummy_input = jnp.ones(input_shape)
    print(module.tabulate(rng, dummy_input, depth=1))


class MLP(nn.Module):
    """A simple pickle-able multi-layer perceptron.

    Args:
        layer_sizes: Sizes of all hidden layers and the output layer.
        activate_final: Whether to apply an activation function to the output.
        bias: Whether to use a bias in the linear layers.
    """

    layer_sizes: Sequence[int]
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        # TODO(vincekurtz): consider using jax control flow here. Note that
        # standard jax control flows (e.g. jax.lax.scan) do not play nicely with
        # flax, see for example https://github.com/google/flax/discussions/1283.
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                layer_size,
                use_bias=self.bias,
                kernel_init=nn.initializers.lecun_uniform(),
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1:
                x = nn.swish(x)
            if i == len(self.layer_sizes) - 1 and self.activate_final:
                x = nn.tanh(x)
        return x
