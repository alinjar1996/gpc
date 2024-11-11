import time
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax
from hydrax.alg_base import SamplingBasedController

from gpc.architectures import ActionSequenceMLP
from gpc.augmented import PACParams, PredictionAugmentedController
from gpc.env import SimulatorState, TrainingEnv

Params = Any


def simulate_episode(
    env: TrainingEnv,
    ctrl: PredictionAugmentedController,
    net: ActionSequenceMLP,
    params: Params,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Starting from a random initial state, run SPC and record training data.

    Args:
        env: The training environment.
        ctrl: The sampling-based controller (augmented with a learned policy).
        net: The policy network.
        params: The policy network parameters.
        rng: The random number generator key.

    Returns:
        y: The observations at each time step.
        U: The optimal actions at each time step.
        best_cost: cost of the best actions at each time step.
        pred_cost: cost of the policy network's actions at each time step.
    """
    rng, ctrl_rng, env_rng = jax.random.split(rng, 3)

    # Set the initial state of the environment
    x = env.init_state(env_rng)

    # Set the initial sampling-based controller parameters
    psi = ctrl.init_params()
    psi = psi.replace(base_params=psi.base_params.replace(rng=ctrl_rng))

    def _scan_fn(carry: Tuple[SimulatorState, PACParams], t: int) -> Tuple:
        """Take step in the training loop."""
        x, psi = carry

        # Generate an action sequence from the learned policy
        y = env.get_observation(x)
        U_pred = net.apply(params, y)

        # Using the predicted action sequence as one of the samples, find an
        # optimal action sequence
        psi = psi.replace(prediction=U_pred)
        psi, rollouts = ctrl.optimize(x.data, psi)
        U_star = ctrl.get_action_sequence(psi)

        # Record the cost of the predicted action sequence relative to the
        # best sample.
        costs = jnp.sum(rollouts.costs[0], axis=1)
        best_cost = jnp.min(costs)
        pred_cost = costs[-1]  # U_pred gets placed at the end of the samples

        # Step the simulation
        x = env.step(x, U_star[0])

        return (x, psi), (y, U_star, best_cost, pred_cost)

    _, (y, U, best_costs, pred_costs) = jax.lax.scan(
        _scan_fn, (x, psi), jnp.arange(env.episode_length)
    )

    return y, U, best_costs, pred_costs


def fit_policy(
    observations: jax.Array,
    action_sequences: jax.Array,
    net: ActionSequenceMLP,
    params: Params,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    rng: jax.Array,
    batch_size: int,
    num_epochs: int,
) -> Tuple[Params, optax.OptState, jax.Array]:
    """Fit the policy network U = NNet(y) to the given data.

    Args:
        observations: The observations y.
        action_sequences: The corresponding target action sequences U.
        net: The policy network.
        params: The policy network parameters.
        optimizer: The optimizer (e.g. Adam).
        opt_state: The optimizer state.
        rng: The random number generator key for shuffling the data.
        batch_size: The batch size.
        num_epochs: The number of epochs.

    Returns:
        The updated policy network parameters.
        The updated optimizer state.
        The loss from the last epoch.
    """
    num_data_points = observations.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    def _loss_fn(params: Params, obs: jax.Array, act: jax.Array) -> jax.Array:
        """Compute the regression loss for the policy network."""
        pred = net.apply(params, obs)
        return jnp.mean(jnp.square(act - pred))

    def _sgd_step(
        params: Params,
        opt_state: optax.OptState,
        obs: jax.Array,
        act: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Perform a gradient descent step on the given batch of data."""
        loss, grad = jax.value_and_grad(_loss_fn)(params, obs, act)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(num_epochs):
        for _ in range(num_batches):
            # Get a random batch of data
            rng, batch_rng = jax.random.split(rng)
            batch_idx = jax.random.randint(
                batch_rng, (batch_size,), 0, num_data_points
            )
            batch_obs = observations[batch_idx]
            batch_act = action_sequences[batch_idx]

            # Do an optimizer step
            params, opt_state, loss = _sgd_step(
                params,
                opt_state,
                batch_obs,
                batch_act,
            )

        print(f"  loss: {loss:.5f}")

    return params, opt_state, loss


def train(env: TrainingEnv, ctrl: SamplingBasedController) -> None:
    """Train a generative predictive controller.

    Args:
        env: The training environment.
        ctrl: The controller to train.
    """
    rng = jax.random.key(0)

    # Set up a sampling-based controller that replaces one of the samples with
    # a control tape prediction from a learned policy.
    ctrl = PredictionAugmentedController(ctrl)
    assert env.task == ctrl.task

    # Set up the policy network that generates action sequences
    # TODO: make the network architecture an argument
    net = ActionSequenceMLP(
        hidden_layers=(32, 32),
        num_steps=env.task.planning_horizon,
        action_dim=env.task.model.nu,
    )
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, jnp.zeros(env.observation_size))

    # Initialize the optimizer
    optimizer = optax.adam(1e-3)  # TODO: set the learning rate as an argument
    opt_state = optimizer.init(params)

    # Gather data
    rng, episode_rng = jax.random.split(rng)
    episode_rngs = jax.random.split(episode_rng, 10)
    st = time.time()
    y, U, best, pred = jax.vmap(
        simulate_episode, in_axes=(None, None, None, None, 0)
    )(env, ctrl, net, params, episode_rngs)
    print(y.shape, U.shape, best.shape, pred.shape)
    print("Time taken:", time.time() - st)

    # Flatten the dataset for training
    y = y.reshape(-1, y.shape[-1])
    U = U.reshape(-1, env.task.planning_horizon, env.task.model.nu)

    # Fit the policy network
    batch_size = 128
    num_epochs = 10  # TODO: set as arguments
    params, opt_state, loss = fit_policy(
        y, U, net, params, optimizer, opt_state, rng, batch_size, num_epochs
    )

    print("Final loss:", loss)
