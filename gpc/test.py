import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from hydrax.task_base import Task

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
    # TODO: read size from task
    actions = jnp.zeros((5, 2))

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            st = time.time()

            # TODO: get observation from the task
            pos = mj_data.qpos[:2] - mj_data.mocap_pos[0, :2]
            obs = jnp.array([pos, mj_data.qvel]).flatten()

            actions = jit_policy(actions, obs)
            mj_data.ctrl[:] = actions[0]

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)
