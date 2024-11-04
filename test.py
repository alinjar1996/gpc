import pickle
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from train import MLP, ScoreMLP

"""
Test a trained model with an interactive sim.
"""


def test_predictive_sampling() -> None:
    """Test the original predictive sampling alg."""
    # Set up the policy
    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)
    policy_params = ctrl.init_params()
    mjx_data = mjx.make_data(task.model)
    jit_policy = jax.jit(
        lambda mjx_data, policy_params: ctrl.optimize(mjx_data, policy_params)[
            0
        ],
        donate_argnames=("policy_params",),
    )

    # Set up the mujoco simultion
    xml_path = ROOT + "/models/particle/scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    mj_data.mocap_pos = np.zeros((1, 3))

    # Set a run time (seconds)
    run_time = 10
    total_cost = 0.0

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and mj_data.time < run_time:
            st = time.time()

            # Set the target position
            x = 0.2 * jnp.cos(1 * mj_data.time)
            y = 0.2 * jnp.sin(2 * mj_data.time)
            mj_data.mocap_pos[0, :] = np.array([x, y, 0])

            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
            )
            policy_params = jit_policy(mjx_data, policy_params)

            mj_data.ctrl[:] = ctrl.get_action(policy_params, 0.0)

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)

            pos = mj_data.qpos[:2] - mj_data.mocap_pos[0, :2]
            total_cost += jnp.linalg.norm(pos)

    print(f"Total cost: {total_cost:.2f}")


def test_simple_policy() -> None:
    """Test a simple RL-style policy that maps observations to actions."""
    # Load the policy from disc
    with open("/tmp/simple_policy.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["net"]
    params = data["params"]
    assert isinstance(model, MLP)

    jit_policy = jax.jit(lambda obs: model.apply(params, obs))

    # Set up the mujoco simultion
    xml_path = ROOT + "/models/particle/scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Set a run time (seconds)
    run_time = 10
    total_cost = 0.0

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and mj_data.time < run_time:
            st = time.time()

            # Set the target position
            x = 0.2 * jnp.cos(1 * mj_data.time)
            y = 0.2 * jnp.sin(2 * mj_data.time)
            mj_data.mocap_pos[0, :] = np.array([x, y, 0])

            pos = mj_data.qpos[:2] - mj_data.mocap_pos[0, :2]
            obs = jnp.array([pos, mj_data.qvel]).flatten()
            mj_data.ctrl[:] = jit_policy(obs)

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)

            total_cost += jnp.linalg.norm(pos)

    print(f"Total cost: {total_cost:.2f}")


def test_gpc_policy() -> None:
    """Test a GPC policy that updates the action sequence."""
    # Load the policy from disc
    with open("/tmp/gpc_policy.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["net"]
    params = data["params"]
    assert isinstance(model, ScoreMLP)

    jit_policy = jax.jit(
        lambda old_actions, obs: model.apply(params, old_actions, obs),
        donate_argnames=("old_actions",),
    )

    # Set up the mujoco simultion
    xml_path = ROOT + "/models/particle/scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Initialize the action sequence
    actions = jnp.zeros((1, 5, 2))

    # Set a run time (seconds)
    run_time = 10
    total_cost = 0.0

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and mj_data.time < run_time:
            st = time.time()

            # Set the target position
            x = 0.2 * jnp.cos(1 * mj_data.time)
            y = 0.2 * jnp.sin(2 * mj_data.time)
            mj_data.mocap_pos[0, :] = np.array([x, y, 0])

            pos = mj_data.qpos[:2] - mj_data.mocap_pos[0, :2]
            obs = jnp.array([pos, mj_data.qvel]).flatten()[None]
            actions = jit_policy(actions, obs)

            mj_data.ctrl[:] = actions[0, 0]

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)

            total_cost += jnp.linalg.norm(pos)

    print(f"Total cost: {total_cost:.2f}")


if __name__ == "__main__":
    test_predictive_sampling()
    # test_simple_policy()
    # test_gpc_policy()
