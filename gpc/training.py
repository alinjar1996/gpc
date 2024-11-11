import time
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax
from hydrax.alg_base import SamplingBasedController

from gpc.architectures import ActionSequenceMLP
from gpc.augmented_controller import PACParams, PredictionAugmentedController
from gpc.env import SimulatorState, TrainingEnv

Params = Any


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

    # Define some helper functions
    def _scan_fn(carry: Tuple[SimulatorState, PACParams], t: int) -> Tuple:
        """Take step in the training loop."""
        x, psi = carry

        # Generate an action sequence from the learned policy
        y = env.get_observation(x)
        U_pred = jnp.zeros((ctrl.task.planning_horizon, ctrl.task.model.nu))

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

    def _run_episode(
        rng: jax.Array, params: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Simulate an episode and collect policy training data.

        Args:
            rng: The random number generator key.
            params: The policy network parameters.

        Returns:
            Observations y.
            Optimal actions U.
            Average cost of the best sample.
            Average cost of the policy's sample.
        """
        rng, ctrl_rng, env_rng = jax.random.split(rng, 3)

        # Set the initial state of the environment
        x = env.init_state(env_rng)

        # Set the initial sampling-based controller parameters
        psi = ctrl.init_params()
        psi = psi.replace(base_params=psi.base_params.replace(rng=ctrl_rng))

        _, (y, U, best_costs, pred_costs) = jax.lax.scan(
            _scan_fn, (x, psi), jnp.arange(env.episode_length)
        )

        return y, U, jnp.mean(best_costs), jnp.mean(pred_costs)

    def _loss_fn(params: Params, obs: jax.Array, act: jax.Array) -> jax.Array:
        """Compute the regression loss for the policy network."""
        pred = net.apply(params, obs)
        return jnp.mean(jnp.square(act - pred))

    loss_and_grad = jax.value_and_grad(_loss_fn)

    def _sgd_step(
        params: Params,
        opt_state: optax.OptState,
        obs: jax.Array,
        act: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Perform a gradient descent step on the given batch of data."""
        loss, grad = loss_and_grad(params, obs, act)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Gather data
    # rng, episode_rng = jax.random.split(rng)
    # episode_rngs = jax.random.split(episode_rng, 10)
    # st = time.time()
    # y, U, best, pred = jax.vmap(_run_episode, in_axes=(0, None))(
    #     episode_rngs, None
    # )
    # print(y.shape, U.shape, best, pred)
    # print(jnp.mean(best), jnp.mean(pred))
    # print("Time taken:", time.time() - st)
    rng, y_rng, U_rng = jax.random.split(rng, 3)
    y = jax.random.normal(y_rng, (10, env.episode_length, 4))  # fake data
    U = 0.1 * jax.random.normal(U_rng, (10, env.episode_length, 5, 2))

    # Flatten the dataset for training
    y = y.reshape(-1, y.shape[-1])
    U = U.reshape(-1, env.task.planning_horizon, env.task.model.nu)
    print(y.shape)
    print(U.shape)

    # Fit the policy network
    num_epochs = 10  # TODO: set as arguments
    batch_size = 128
    num_data_points = y.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    st = time.time()
    for e in range(num_epochs):
        for _ in range(num_batches):
            # Get a random batch of data
            rng, batch_rng = jax.random.split(rng)
            batch_idx = jax.random.randint(
                batch_rng, (batch_size,), 0, num_data_points
            )
            batch_obs = y[batch_idx]
            batch_act = U[batch_idx]

            # Do an optimizer step
            params, opt_state, loss = _sgd_step(
                params,
                opt_state,
                batch_obs,
                batch_act,
            )

        # TODO: more systematic logging
        print(
            f"  epoch {e+1}/{num_epochs}, loss: {loss:.5f}, "
            f"time: {time.time() - st:.2f} s"
        )
