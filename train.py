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


class ScoreMLP(nn.Module):
    """A pickle-able conditional score network for generative models.

        dx = s(x | y)

    Args:
        hidden_layers: Sizes of all hidden layers.
        bias: Whether to use a bias in the linear layers.
    """

    hidden_layers: Sequence[int]
    bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Forward pass through the network."""
        # TODO: consider setting these as parameters
        data_dim = x.shape[-1]
        traj_len = x.shape[-2]

        # Concatenate the input and conditioning data
        x_flat = x.reshape(-1, data_dim * traj_len)
        x = jnp.concatenate([x_flat, y], axis=-1)

        for i, layer_size in enumerate(self.hidden_layers):
            x = nn.Dense(
                layer_size,
                use_bias=self.bias,
                kernel_init=nn.initializers.lecun_uniform(),
                name=f"dense_{i}",
            )(x)
            x = nn.swish(x)

        x = nn.Dense(
            data_dim * traj_len,
            use_bias=self.bias,
            kernel_init=nn.initializers.lecun_uniform(),
            name=f"dense_{len(self.hidden_layers)}",
        )(x)

        return x.reshape(-1, traj_len, data_dim)


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

    # Save the model
    fname = "/tmp/simple_policy.pkl"
    with open(fname, "wb") as f:
        save_dict = {"net": model, "params": params}
        pickle.dump(save_dict, f)
    print(f"Model saved to {fname}")


def train_gpc(data: TrainingData, rng: jax.Array) -> None:
    """Train a generative model policy, which updates the action sequence."""
    # Set up the inputs and outputs (old_actions, obs --> new_actions)
    act_dim = data.new_action_sequence.shape[-1]
    traj_len = data.new_action_sequence.shape[-2]
    obs_dim = data.obs.shape[-1]
    old_actions = data.old_action_sequence.reshape(-1, traj_len, act_dim)
    new_actions = data.new_action_sequence.reshape(-1, traj_len, act_dim)
    obs = data.obs.reshape(-1, obs_dim)

    N = old_actions.shape[0]
    assert new_actions.shape[0] == N
    assert obs.shape[0] == N

    # Initialize the model
    rng, init_rng = jax.random.split(rng)
    hidden_layers = [64, 64]
    model = ScoreMLP(hidden_layers)
    params = model.init(init_rng, old_actions[0:1], obs[0:1])

    # Define the loss function
    def loss_fn(
        params: Params,
        old_actions: jax.Array,
        obs: jax.Array,
        new_actions: jax.Array,
    ) -> jax.Array:
        pred = model.apply(params, old_actions, obs)
        return jnp.mean((pred - new_actions) ** 2)

    loss_and_grad = jax.value_and_grad(loss_fn)

    # Set up the optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_fn(
        params: Params,
        opt_state: optax.OptState,
        old_actions: jax.Array,
        obs: jax.Array,
        new_actions: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Update the model parameters."""
        loss, grad = loss_and_grad(params, old_actions, obs, new_actions)
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
            batch_old_actions = old_actions[batch_idx]
            batch_obs = obs[batch_idx]
            batch_new_actions = new_actions[batch_idx]

            # Do an optimizer step
            params, opt_state, loss = update_fn(
                params,
                opt_state,
                batch_old_actions,
                batch_obs,
                batch_new_actions,
            )

        print(
            f"Epoch {e+1}/{epochs}, Loss: {loss:.4f}, "
            f"Time: {time.time() - st:.2f} s"
        )

    # Save the model
    fname = "/tmp/gpc_policy.pkl"
    with open(fname, "wb") as f:
        save_dict = {"net": model, "params": params}
        pickle.dump(save_dict, f)
    print(f"Model saved to {fname}")


if __name__ == "__main__":
    # Load the dataset
    with open("/tmp/gpc_particle_data.pkl", "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, TrainingData)

    rng = jax.random.key(0)
    rng, train_rng = jax.random.split(rng)

    # Train a simple RL-style policy
    train_simple_policy(data, train_rng)

    # Train a generative model policy
    # train_gpc(data, train_rng)
