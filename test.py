import pickle
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from hydrax import ROOT

from train import MLP, ScoreMLP

"""
Test a trained model with an interactive sim.
"""


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

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            st = time.time()

            pos = mj_data.qpos[:2] - mj_data.mocap_pos[0, :2]
            obs = jnp.array([pos, mj_data.qvel]).flatten()
            mj_data.ctrl[:] = jit_policy(obs)

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)


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

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            st = time.time()

            pos = mj_data.qpos[:2] - mj_data.mocap_pos[0, :2]
            obs = jnp.array([pos, mj_data.qvel]).flatten()[None]
            actions = jit_policy(actions, obs)

            mj_data.ctrl[:] = actions[0, 0]

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)


if __name__ == "__main__":
    # test_simple_policy()
    test_gpc_policy()
