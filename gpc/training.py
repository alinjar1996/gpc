import pickle
import time
from pathlib import Path
from typing import Any, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydrax.task_base import Task

from gpc.dataset import TrainingData

Params = Any


class Policy:
    """A pickle-able Generative Predictive Control policy."""

    def __init__(
        self, net: nn.Module, params: Any, u_min: jax.Array, u_max: jax.Array
    ):
        """Create a new GPC policy.

        Args:
            net: The network that maps action sequence and observation to a new
                 action sequence.
            params: The parameters of the network.
            u_min: The minimum action values.
            u_max: The maximum action values.
        """
        self.net = net
        self.params = params
        self.u_min = u_min
        self.u_max = u_max

    def save(self, path: Union[Path, str]) -> None:
        """Save the policy to a file.

        Args:
            path: The path to save the policy to.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "Policy":
        """Load a policy from a file.

        Args:
            path: The path to load the policy from.

        Returns:
            The loaded policy instance
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def apply(self, u: jax.Array, y: jax.Array) -> jax.Array:
        """Apply the policy, updating the action sequence.

        Args:
            u: The current action sequence.
            y: The current observation.

        Returns:
            The updated action sequence
        """
        u_new = self.net.apply(self.params, u, y)
        return jax.numpy.clip(u_new, self.u_min, self.u_max)


def train(
    data: TrainingData,
    task: Task,
    net: nn.Module,
    batch_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    print_every: int = 10,
) -> Policy:
    """Train a GPC policy based on the given training data.

    Args:
        data: The training data to use.
        task: The task to train the policy for.
        net: Network architecture, U_new = NNet(U_old, y).
        batch_size: The number of data points in each batch.
        epochs: The number of epochs (passes over the whole dataset).
        learning_rate: The learning rate for the Adam optimizer.
        print_every: The number of epochs between printing progress data.

    Returns:
        The trained policy.
    """
    rng = jax.random.key(0)  # TODO: take as argument

    # Determine the action and observation dimensions
    act_dim = data.new_action_sequence.shape[-1]
    obs_dim = data.observation.shape[-1]
    horizon = data.new_action_sequence.shape[-2]

    # Reshape the data to flatten across initial states
    # TODO: shuffle and train-test split
    old_actions = 0.0 * data.old_action_sequence.reshape(-1, horizon, act_dim)
    new_actions = data.new_action_sequence.reshape(-1, horizon, act_dim)
    obs = data.observation.reshape(-1, obs_dim)

    num_data_points = old_actions.shape[0]
    assert new_actions.shape[0] == num_data_points
    assert obs.shape[0] == num_data_points

    # Initialize the model
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, old_actions[0], obs[0])

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Define loss and parameter update functions
    def loss_fn(
        params: Params,
        old_actions: jax.Array,
        obs: jax.Array,
        new_actions: jax.Array,
    ) -> jax.Array:
        """Loss function drives network to predict the new actions."""
        pred = net.apply(params, old_actions, obs)
        return jnp.mean((pred - new_actions) ** 2)

    loss_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def update_fn(
        params: Params,
        opt_state: optax.OptState,
        old_actions: jax.Array,
        obs: jax.Array,
        new_actions: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Perform a gradient descent step."""
        loss, grad = loss_and_grad(params, old_actions, obs, new_actions)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Train the model
    num_batches = num_data_points // batch_size
    st = time.time()
    for e in range(epochs):
        for _ in range(num_batches):
            # Get a random batch of data
            rng, batch_rng = jax.random.split(rng)
            batch_idx = jax.random.randint(
                batch_rng, (batch_size,), 0, num_data_points
            )
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

        # TODO: more systematic logging
        if (e + 1) % print_every == 0:
            print(
                f"  epoch {e+1}/{epochs}, loss: {loss:.5f}, "
                f"time: {time.time() - st:.2f} s"
            )

    # Construct the policy
    return Policy(net, params, task.u_min, task.u_max)
