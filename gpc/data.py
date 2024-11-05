import time
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task
from mujoco import mjx

"""
Tools for collecting data by running closed-loop simulations.
"""

ResetFn = Callable[[int, jax.Array], jax.Array]


@dataclass
class TrainingData:
    """Container for training data collected during simulation.

    Attributes:
        observation: The current observation.
        old_action_sequence: The previous action sequence.
        new_action_sequence: The updated action sequence.
    """

    observation: jax.Array
    old_action_sequence: jax.Array
    new_action_sequence: jax.Array


def collect_data(
    task: Task,
    ctrl: SamplingBasedController,
    num_timesteps: int,
    num_resets: int,
    reset_fn: Callable[[mjx.Data, jax.Array], mjx.Data],
    rng: jax.Array,
) -> TrainingData:
    """Collect training data by running closed-loop simulations.

    Args:
        task: The task to simulate.
        ctrl: The controller to use for data collection.
        num_timesteps: The number of timesteps to simulate per rollout.
        num_resets: The number of rollouts to collect from different states.
        reset_fn: The function to reset the simulator to a new initial state.
        rng: The random number generator key.

    Returns:
        The collected training data, shape (num_resets, num_timesteps, ...).
    """
    policy_params = ctrl.init_params()
    jit_reset = jax.jit(reset_fn, donate_argnums=(0,))

    # Set up the simulator (which matches the controller's model exactly)
    mjx_model = task.model
    mjx_data = mjx.make_data(mjx_model)

    def _get_action_sequence(policy_params: jax.Array) -> jax.Array:
        """Get the action sequence from the controller."""
        timesteps = jnp.arange(task.planning_horizon) * task.dt
        return jax.vmap(ctrl.get_action, in_axes=(None, 0))(
            policy_params, timesteps
        )

    def _collect_one_rollout(
        rng: jax.Array, mjx_data: mjx.Data
    ) -> TrainingData:
        """Collect data from one rollout."""
        rng, reset_rng = jax.random.split(rng)
        mjx_data = jit_reset(mjx_data, reset_rng)

        def _step(carry: Tuple, t: int) -> Tuple[Tuple, TrainingData]:
            """Simulate one step of the controller."""
            policy_params, mjx_data = carry

            # Update the policy parameters
            old_actions = _get_action_sequence(policy_params)
            policy_params, rollouts = ctrl.optimize(mjx_data, policy_params)
            new_actions = _get_action_sequence(policy_params)

            # Observations are sorted by randomization, rollout, time
            obs = rollouts.observations[0, 0, 0]

            # Apply the first control action
            u = ctrl.get_action(policy_params, 0.0)
            mjx_data = mjx.step(mjx_model, mjx_data.replace(ctrl=u))

            return (policy_params, mjx_data), TrainingData(
                observation=obs,
                old_action_sequence=old_actions,
                new_action_sequence=new_actions,
            )

        _, data = jax.lax.scan(
            _step, (policy_params, mjx_data), jnp.arange(num_timesteps)
        )
        return data

    reset_rngs = jax.random.split(rng, num_resets)
    data = jax.vmap(_collect_one_rollout, in_axes=(0, None))(
        reset_rngs, mjx_data
    )
    return data


def visualize_data(
    task: Task,
    data: TrainingData,
) -> None:
    """Play back the training data on the mujoco visualizer."""
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    num_resets, num_timesteps, _ = data.observation.shape

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        i = 0
        while viewer.is_running():
            print(f"  {i+1}/{num_resets}...", end="\r")
            for t in range(num_timesteps):
                st = time.time()

                # TODO: generalize observation to mj_data
                mj_data.qpos[:] = data.observation[i, t, :2]
                mj_data.qvel[:] = data.observation[i, t, 2:]

                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                elapsed = time.time() - st
                if elapsed < mj_model.opt.timestep:
                    time.sleep(mj_model.opt.timestep - elapsed)

            i += 1
            if i == num_resets:
                i = 0
