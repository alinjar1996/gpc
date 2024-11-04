import pickle
import time
from typing import Any, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from collect_data import TrainingData

"""
Train a generative model based on collected sim data.
"""

Params = Any


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


def train_simple_policy(data: TrainingData, rng: jax.Array) -> None:
    """Train a simple RL-style policy that maps observations to actions."""
    # Set up the inputs and outputs
    inputs = data.obs.reshape(-1, data.obs.shape[-1])
    applied_action = data.new_action_sequence[:, :, 0, :]
    targets = applied_action.reshape(-1, applied_action.shape[-1])

    N = inputs.shape[0]
    assert targets.shape[0] == N

    # Initialize the model
    rng, init_rng = jax.random.split(rng)
    layer_sizes = [64, 64, 2]
    model = MLP(layer_sizes)
    params = model.init(init_rng, jnp.zeros((1, 4)))

    # Define the loss function
    def loss_fn(
        params: Params, inputs: jax.Array, targets: jax.Array
    ) -> jax.Array:
        pred = model.apply(params, inputs)
        return jnp.mean((pred - targets) ** 2)

    loss_and_grad = jax.value_and_grad(loss_fn)

    # Set up the optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_fn(
        params: Params,
        opt_state: optax.OptState,
        inputs: jax.Array,
        targets: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Update the model parameters."""
        loss, grad = loss_and_grad(params, inputs, targets)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Train the model
    batch_size = 128
    num_batches = N // batch_size
    epochs = 100

    st = time.time()
    for e in range(epochs):
        for _ in range(num_batches):
            # Get a random batch of data
            rng, batch_rng = jax.random.split(rng)
            batch_idx = jax.random.randint(batch_rng, (batch_size,), 0, N)
            batch_inputs = inputs[batch_idx]
            batch_targets = targets[batch_idx]

            # Do an optimizer step
            params, opt_state, loss = update_fn(
                params, opt_state, batch_inputs, batch_targets
            )

        print(
            f"Epoch {e+1}/{epochs}, Loss: {loss:.4f}, "
            f"Time: {time.time() - st:.2f} s"
        )


if __name__ == "__main__":
    # Load the dataset
    with open("/tmp/gpc_particle_data.pkl", "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, TrainingData)

    # Train a simple RL-style policy
    rng = jax.random.key(0)
    rng, train_rng = jax.random.split(rng)
    train_simple_policy(data, train_rng)
