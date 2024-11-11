import time
from pathlib import Path
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from hydrax.alg_base import SamplingBasedController
from tensorboardX import SummaryWriter

from gpc.architectures import ActionSequenceMLP
from gpc.augmented import PACParams, PredictionAugmentedController
from gpc.envs import SimulatorState, TrainingEnv
from gpc.policy import Policy

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
        J_best: cost of the best actions at each time step.
        J_pred: cost of the policy network's actions at each time step.
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

    _, (y, U, J_best, J_pred) = jax.lax.scan(
        _scan_fn, (x, psi), jnp.arange(env.episode_length)
    )

    return y, U, J_best, J_pred


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

    def _scan(carry: Tuple[Params, optax.OptState, jax.Array], t: int) -> Tuple:
        """Inner loop function for the optimizer."""
        params, opt_state, rng = carry

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

        return (params, opt_state, rng), loss

    (params, opt_state, rng), losses = jax.lax.scan(
        _scan, (params, opt_state, rng), jnp.arange(num_epochs * num_batches)
    )

    return params, opt_state, losses[-1]


def train(
    env: TrainingEnv,
    ctrl: SamplingBasedController,
    net: ActionSequenceMLP,
    log_dir: Union[Path, str],
    num_iters: int,
    num_envs: int,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    num_epochs: int = 10,
) -> None:
    """Train a generative predictive controller.

    Args:
        env: The training environment.
        ctrl: The sampling-based predictive control method to use.
        net: The policy network architecture, maps observation to control tape.
        log_dir: The directory to log TensorBoard data to.
        num_iters: The number of training iterations.
        num_envs: The number of parallel environments to simulate.
        learning_rate: The learning rate for the policy network.
        batch_size: The batch size for training the policy network.
        num_epochs: The number of epochs to train the policy network.

    Note that the total number of parallel simulations is
        `num_envs * ctrl.num_samples * ctrl.num_randomizations`

    """
    rng = jax.random.key(0)

    # Set up the sampling-based controller and policy network
    ctrl = PredictionAugmentedController(ctrl)
    assert env.task == ctrl.task

    # Set up the policy network
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, jnp.zeros(env.observation_size))
    U_test = net.apply(params, jnp.zeros(env.observation_size))
    assert U_test.shape == (env.task.planning_horizon, env.task.model.nu)

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Set up the TensorBoard logger
    datetime = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir) / datetime
    print("Logging to", log_dir)
    tb_writer = SummaryWriter(log_dir)

    # Set up some helper functions
    @jax.jit
    def jit_simulate(
        params: Params, rng: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Simulate episodes in parallel.

        Args:
            params: The policy network parameters.
            rng: The random number generator key.

        Returns:
            y: The observations at each time step.
            U: The optimal actions at each time step.
            Average cost of the best action sequence.
            Average cost of the policy's predicted action sequence.
        """
        rngs = jax.random.split(rng, num_envs)

        y, U, J_best, J_pred = jax.vmap(
            simulate_episode, in_axes=(None, None, None, None, 0)
        )(env, ctrl, net, params, rngs)
        return y, U, jnp.mean(J_best), jnp.mean(J_pred)

    @jax.jit
    def jit_fit(
        observations: jax.Array,
        actions: jax.Array,
        params: Params,
        opt_state: optax.OptState,
        rng: jax.Array,
    ) -> Tuple[Params, optax.OptState, jax.Array]:
        """Fit the policy network to the data.

        Args:
            observations: The observations.
            actions: The best action sequences.
            params: The policy network parameters.
            opt_state: The optimizer state.
            rng: The random number generator key.

        Returns:
            The updated policy network parameters.
            The updated optimizer state.
            The loss from the last epoch.
        """
        y = observations.reshape(-1, observations.shape[-1])
        U = actions.reshape(-1, env.task.planning_horizon, env.task.model.nu)
        return fit_policy(
            y, U, net, params, optimizer, opt_state, rng, batch_size, num_epochs
        )

    for i in range(num_iters):
        # Simulate using SPC and record the best action sequences. One of the
        # samples gets replaced with the policy output U = NNet(y).
        sim_start = time.time()
        rng, episode_rng = jax.random.split(rng)
        y, U, J_best, J_pred = jit_simulate(params, episode_rng)
        sim_time = time.time() - sim_start

        # Fit the policy network U = NNet(y) to the data
        fit_start = time.time()
        rng, fit_rng = jax.random.split(rng)
        params, opt_state, loss = jit_fit(y, U, params, opt_state, fit_rng)
        fit_time = time.time() - fit_start

        # TODO: run some evaluation tests

        # Print a performance summary
        print(
            f"  {i+1}/{num_iters} |"
            f" policy: {J_pred:.4f} |"
            f" best: {J_best:.4f} |"
            f" loss: {loss:.4f} |"
            f" iter time: {sim_time + fit_time:.4f} s"
        )

        # Tensorboard logging
        tb_writer.add_scalar("sim/policy_cost", J_pred, i)
        tb_writer.add_scalar("sim/best_cost", J_best, i)
        tb_writer.add_scalar("sim/time", sim_time, i)
        tb_writer.add_scalar("fit/loss", loss, i)
        tb_writer.add_scalar("fit/time", fit_time, i)
        tb_writer.flush()

    # Create a pickle-able policy object
    policy = Policy(net, params, env.task.u_min, env.task.u_max)
    return policy
