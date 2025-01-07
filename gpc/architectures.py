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
        self.hidden_layers = hidden_layers

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


class Conv1DBlock(nnx.Module):
    """A simple temporal convolutional block.

         ----------     -------------     --------------------
    ---> | Conv1d | --> | BatchNorm | --> | Swish Activation | --->
         ----------     -------------     --------------------

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        rngs: nnx.Rngs,
    ):
        """Initialize the block.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            kernel_size: Size of the convolutional kernel.
            rngs: Random number generators for initialization.
        """
        self.c = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(num_features=out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the block."""
        x = self.c(x)
        x = self.bn(x)
        x = nnx.swish(x)
        return x


class ConditionalResidualBlock(nnx.Module):
    """A temporal convolutional block with FiLM conditional information.

        -------------------------------------------------------------
        |                                                           |
        |  -----------             -----------     -----------      |
    x ---> | Encoder | --> (+) --> | Dropout | --> | Decoder | --> (+) -->
           -----------      |      -----------     -----------
                            |
                       ----------
    y -----------------| Linear |
                       ----------

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cond_features: int,
        kernel_size: int,
        rngs: nnx.Rngs,
    ):
        """Initialize the block.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            cond_features: Number of conditioning features.
            kernel_size: Size of the convolutional kernel.
            rngs: Random number generators for initialization.
        """
        self.encoder = Conv1DBlock(in_features, out_features, kernel_size, rngs)
        self.decoder = Conv1DBlock(
            out_features, out_features, kernel_size, rngs
        )
        self.linear = nnx.LinearGeneral(
            cond_features, (1, out_features), rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.residual = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Forward pass through the block."""
        z = self.encoder(x)
        z += self.linear(y)
        z = self.dropout(z)
        z = self.decoder(z)
        return z + self.residual(x)


class DenoisingCNN(nnx.Module):
    """A denoising convolutional network with FiLM conditioning.

    Based on Diffusion Policy, https://arxiv.org/abs/2303.04137v5.
    """

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        horizon: int,
        feature_dims: Sequence[int],
        rngs: nnx.Rngs,
    ):
        """Initialize the network.

        Args:
            action_size: Dimension of the actions (u).
            observation_size: Dimension of the observations (y).
            horizon: Number of steps in the action sequence (U = [u0, u1, ...]).
            feature_dims: List of feature dimensions.
            rngs: Random number generators for initialization.
        """
        self.action_size = action_size
        self.observation_size = observation_size
        self.horizon = horizon
        self.num_layers = len(feature_dims) + 1
        kernel_size = 3

        feature_sizes = [action_size] + list(feature_dims) + [action_size]
        for i, (input_size, output_size) in enumerate(
            zip(feature_sizes[:-1], feature_sizes[1:], strict=False)
        ):
            setattr(
                self,
                f"l{i}",
                ConditionalResidualBlock(
                    input_size,
                    output_size,
                    observation_size + 1,
                    kernel_size,
                    rngs,
                ),
            )

    def __call__(self, u: jax.Array, y: jax.Array, t: jax.Array) -> jax.Array:
        """Forward pass through the network."""
        y = jnp.concatenate([y, t], axis=-1)

        x = self.l0(u, y)
        for i in range(1, self.num_layers):
            x = getattr(self, f"l{i}")(x, y)

        return x + u
