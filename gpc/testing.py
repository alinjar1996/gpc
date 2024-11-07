import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from hydrax.task_base import Task
from mujoco import mjx

from gpc.training import Policy


def test_interactive(task: Task, policy: Policy) -> None:
    """Test a GPC policy with an interactive simulation.

    Args:
        task: The task to run the policy on.
        policy: The GPC policy to test.
    """
    jit_policy = jax.jit(policy.apply, donate_argnums=(0,))

    # Set up the mujoco simultion
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Initialize the action sequence
    actions = jnp.zeros((task.planning_horizon, task.model.nu))

    # Set up an observation function
    mjx_data = mjx.make_data(task.model)

    @jax.jit
    def get_obs(mjx_data: mjx.Data) -> jax.Array:
        """Get an observation from the mujoco data."""
        mjx_data = mjx.forward(task.model, mjx_data)  # update sites & sensors
        return task.get_obs(mjx_data)

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            st = time.time()

            # Get an observation
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
            )
            obs = get_obs(mjx_data)

            # Update the action sequence
            inference_start = time.time()
            actions = jit_policy(actions, obs)
            mj_data.ctrl[:] = actions[0]

            obs_time = inference_start - st
            inference_time = time.time() - inference_start
            print(
                f"  Observation time: {obs_time:.5f}s "
                f" Inference time: {inference_time:.5f}s",
                end="\r",
            )

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)
