import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from hydrax.alg_base import SamplingBasedController
from tensorboardX import SummaryWriter

from gpc.architectures import DenoisingMLP
from gpc.augmented import PACParams, PolicyAugmentedController
from gpc.envs import SimulatorState, TrainingEnv
from gpc.policy import Policy

Params = Any


def simulate_episode(
    env: TrainingEnv,
    ctrl: PolicyAugmentedController,
    policy: Policy,
    exploration_noise_level: float,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Starting from a random initial state, run SPC and record training data.

    Args:
        env: The training environment.
        ctrl: The sampling-based controller (augmented with a learned policy).
        policy: The generative policy network.
        exploration_noise_level: Standard deviation of the gaussian noise added
                                 to each action.
        rng: The random number generator key.

    Returns:
        y: The observations at each time step.
        U: The optimal actions at each time step.
        J_spc: cost of the best action sequence found by SPC at each time step.
        J_policy: cost of the best action sequence found by the policy.
    """
    rng, ctrl_rng, env_rng = jax.random.split(rng, 3)

    # Set the initial state of the environment
    x = env.init_state(env_rng)

    # Set the initial sampling-based controller parameters
    psi = ctrl.init_params()
    psi = psi.replace(base_params=psi.base_params.replace(rng=ctrl_rng))

    def _scan_fn(carry: Tuple[SimulatorState, PACParams], t: int) -> Tuple:
        """Take simulation step, and record all data."""
        x, psi = carry

        # Sample action sequences from the learned policy
        # TODO: consider warm-starting the policy
        y = env._get_observation(x)
        rng, policy_rng, explore_rng = jax.random.split(psi.base_params.rng, 3)
        policy_rngs = jax.random.split(policy_rng, ctrl.num_policy_samples)
        U = jnp.zeros((env.task.planning_horizon, env.task.model.nu))
        Us = jax.vmap(policy.apply, in_axes=(None, None, 0))(U, y, policy_rngs)

        # Place the samples into the predictive control parameters so they
        # can be used in the predictive control update
        psi = psi.replace(
            policy_samples=Us, base_params=psi.base_params.replace(rng=rng)
        )

        # Update the action sequence with sampling-based predictive control
        psi, rollouts = ctrl.optimize(x.data, psi)
        U_star = ctrl.get_action_sequence(psi)

        # Record the lowest costs achieved by SPC and the policy
        # TODO: consider logging something more informative
        costs = jnp.sum(rollouts.costs, axis=1)
        spc_best = jnp.min(costs[: -ctrl.num_policy_samples])
        policy_best = jnp.min(costs[ctrl.num_policy_samples :])

        # Step the simulation
        exploration_noise = exploration_noise_level * jax.random.normal(
            explore_rng, U_star[0].shape
        )
        x = env.step(x, U_star[0] + exploration_noise)

        return (x, psi), (y, U_star, spc_best, policy_best)

    _, (y, U, J_spc, J_policy) = jax.lax.scan(
        _scan_fn, (x, psi), jnp.arange(env.episode_length)
    )

    return y, U, J_spc, J_policy


def fit_policy(
    observations: jax.Array,
    action_sequences: jax.Array,
    model: DenoisingMLP,
    optimizer: nnx.Optimizer,
    batch_size: int,
    num_epochs: int,
    rng: jax.Array,
) -> jax.Array:
    """Fit a flow matching model to the data.

    Args:
        observations: The observations y.
        action_sequences: The corresponding target action sequences U.
        model: The policy network, outputs the flow matching vector field.
        optimizer: The optimizer (e.g. Adam).
        batch_size: The batch size.
        num_epochs: The number of epochs.
        rng: The random number generator key.

    Returns:
        The loss from the last epoch.

    Note that model and optimizer are updated in-place by flax.nnx.
    """
    num_data_points = observations.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    def _loss_fn(
        model: nnx.Module,
        obs: jax.Array,
        act: jax.Array,
        noise: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Compute the flow-matching loss."""
        noised_action = t[..., None] * act + (1 - t[..., None]) * noise
        target = act - noise
        pred = model(noised_action, obs, t)
        return jnp.mean(jnp.square(pred - target))

    @nnx.jit
    def _train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Perform a gradient descent step on a batch of data."""
        # Get a random batch of data
        rng, batch_rng = jax.random.split(rng)
        batch_idx = jax.random.randint(
            batch_rng, (batch_size,), 0, num_data_points
        )
        batch_obs = observations[batch_idx]
        batch_act = action_sequences[batch_idx]

        # Sample noise and time steps for the flow matching targets
        rng, noise_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, batch_act.shape)
        t = jax.random.uniform(
            t_rng,
            (
                batch_size,
                1,
            ),
        )

        # Compute the loss and its gradient
        loss, grad = nnx.value_and_grad(_loss_fn)(
            model, batch_obs, batch_act, noise, t
        )

        # Update the optimizer and model parameters in-place via flax.nnx
        optimizer.update(grad)

        return rng, loss

    for _ in range(num_batches * num_epochs):
        rng, loss = _train_step(model, optimizer, rng)

    return loss


def train(  # noqa: PLR0915 this is a long function, don't limit to 50 lines
    env: TrainingEnv,
    ctrl: SamplingBasedController,
    net: DenoisingMLP,
    num_policy_samples: int,
    log_dir: Union[Path, str],
    num_iters: int,
    num_envs: int,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    num_epochs: int = 10,
    checkpoint_every: int = 10,
    exploration_noise_level: float = 0.0,
) -> None:
    """Train a generative predictive controller.

    Args:
        env: The training environment.
        ctrl: The sampling-based predictive control method to use.
        net: The flow matching network architecture.
        num_policy_samples: The number of samples to draw from the policy.
        log_dir: The directory to log TensorBoard data to.
        num_iters: The number of training iterations.
        num_envs: The number of parallel environments to simulate.
        learning_rate: The learning rate for the policy network.
        batch_size: The batch size for training the policy network.
        num_epochs: The number of epochs to train the policy network.
        checkpoint_every: Number of iterations between policy checkpoint saves.
        exploration_noise_level: Standard deviation of the gaussian noise added
                                 to each action during episode simulation.

    """
    rng = jax.random.key(0)

    # Print some information about the training setup
    episode_seconds = env.episode_length * env.task.model.opt.timestep
    horizon_seconds = env.task.planning_horizon * env.task.dt
    num_samples = num_policy_samples + ctrl.num_samples
    print("Training with:")
    print(
        f"  episode length: {episode_seconds} seconds"
        f" ({env.episode_length} simulation steps)"
    )
    (
        print(
            f"  planning horizon: {horizon_seconds} seconds"
            f" ({env.task.planning_horizon} knots)"
        ),
    )
    print(
        "  Parallel rollouts per simulation step:"
        f" {num_samples * ctrl.num_randomizations * num_envs}"
        f" (= {num_samples} x {ctrl.num_randomizations} x {num_envs})"
    )
    print("")

    # Set up the sampling-based controller and policy network
    ctrl = PolicyAugmentedController(ctrl, num_policy_samples)
    assert env.task == ctrl.task

    # Set up the policy
    policy = Policy(net, env.task.u_min, env.task.u_max)

    # Split the policy network into architecture and parameters
    graphdef, params = nnx.split(net)

    # Set up the optimizer
    optimizer = nnx.Optimizer(net, optax.adam(learning_rate))

    # Set up the TensorBoard logger
    log_dir = Path(log_dir) / time.strftime("%Y%m%d_%H%M%S")
    print("Logging to", log_dir)
    tb_writer = SummaryWriter(log_dir)

    # Set up some helper functions
    @nnx.jit
    def jit_simulate(
        policy: Policy, rng: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Simulate episodes in parallel.

        Args:
            params: The policy network parameters.
            rng: The random number generator key.

        Returns:
            The observations at each time step.
            The best action sequence at each time step.
            Average cost of SPC's best action sequence.
            Average cost of the policy's best action sequence.
            Fraction of times the policy generated the best action sequence.
        """
        rngs = jax.random.split(rng, num_envs)

        # TODO: consider making a policy = policy.set_params(params) method
        # net = nnx.merge(graphdef, params)
        # policy = Policy(net, env.task.u_min, env.task.u_max)

        y, U, J_spc, J_policy = jax.vmap(
            simulate_episode, in_axes=(None, None, None, None, 0)
        )(env, ctrl, policy, exploration_noise_level, rngs)

        frac = jnp.mean(J_policy < J_spc)
        return y, U, jnp.mean(J_spc), jnp.mean(J_policy), frac

    @jax.jit
    def jit_fit(
        observations: jax.Array,
        actions: jax.Array,
        net: DenoisingMLP,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
    ) -> Tuple[DenoisingMLP, nnx.Optimizer, jax.Array]:
        """Fit the policy network to the data.

        Args:
            observations: The observations.
            actions: The best action sequences.
            net: The policy network parameters.
            opt_state: The optimizer state.
            rng: The random number generator key.

        Returns:
            The updated policy network parameters.
            The updated optimizer state.
            The loss from the last epoch.
        """
        y = observations.reshape(-1, observations.shape[-1])
        U = actions.reshape(-1, env.task.planning_horizon, env.task.model.nu)
        return fit_policy(y, U, net, optimizer, batch_size, num_epochs, rng)

    train_start = datetime.now()
    for i in range(num_iters):
        # Simulate and record the best action sequences. Some of the action
        # samples are generated via SPC and others are generated by the policy.
        sim_start = time.time()
        rng, episode_rng = jax.random.split(rng)
        y, U, J_spc, J_policy, frac = jit_simulate(policy, episode_rng)
        sim_time = time.time() - sim_start

        breakpoint()

        # Fit the policy network U = NNet(y) to the data
        fit_start = time.time()
        rng, fit_rng = jax.random.split(rng)
        params, opt_state, loss = jit_fit(y, U, params, opt_state, fit_rng)
        fit_time = time.time() - fit_start

        # TODO: run some evaluation tests

        # Save a policy checkpoint
        if i % checkpoint_every == 0 and i > 0:
            ckpt_path = log_dir / f"policy_ckpt_{i}.pkl"
            policy.replace(params=params).save(ckpt_path)
            print(f"Saved policy checkpoint to {ckpt_path}")

        # Print a performance summary
        time_elapsed = datetime.now() - train_start
        print(
            f"  {i+1}/{num_iters} |"
            f" policy cost {J_policy:.4f} |"
            f" spc cost {J_spc:.4f} |"
            f" {100 * frac:.2f}% policy is best |"
            f" loss {loss:.4f} |"
            f" {time_elapsed} elapsed"
        )

        # Tensorboard logging
        tb_writer.add_scalar("sim/policy_cost", J_policy, i)
        tb_writer.add_scalar("sim/spc_cost", J_spc, i)
        tb_writer.add_scalar("sim/time", sim_time, i)
        tb_writer.add_scalar("sim/policy_best_frac", frac, i)
        tb_writer.add_scalar("fit/loss", loss, i)
        tb_writer.add_scalar("fit/time", fit_time, i)
        tb_writer.flush()

    return policy.replace(params=params)
