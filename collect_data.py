import pickle
import time
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from flax.struct import dataclass
from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

"""
Collect training data by running the particle tracking task.
"""


@dataclass
class TrainingData:
    """Data collected during simulation.

    Attributes:
        observation: The current observation.
        old_action_sequence: The previous action sequence.
        new_action_sequence: The new action sequence.
        cost: The cost of the new action sequence.
    """

    obs: jax.Array
    old_action_sequence: jax.Array
    new_action_sequence: jax.Array
    cost: jax.Array


@partial(jax.vmap, in_axes=(None, 0))
def collect_data(num_steps: int, rng: jax.Array) -> TrainingData:
    """Collect training data by running the particle tracking task.

    Args:
        num_steps: The number of steps to simulate.
        rng: The random number generator key for setting the initial conditions.

    Returns:
        The collected training data.
    """
    # Set up the task and control algorithm
    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)

    # Set up the optimizer
    policy_params = ctrl.init_params()

    # Set up the simulator (which matches the controller's model exactly)
    mjx_model = task.model
    mjx_data = mjx.make_data(mjx_model)

    # Reset to a random initial state
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    qpos = jax.random.uniform(pos_rng, (2,), minval=-0.29, maxval=0.29)
    qvel = jax.random.uniform(vel_rng, (2,), minval=-0.5, maxval=0.5)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

    def _step(carry: Tuple, t: int) -> Tuple[Tuple, TrainingData]:
        """Simulate one step of the controller."""
        policy_params, mjx_data = carry

        # Update the policy parameters
        old_actions = policy_params.mean  # TODO: generalize beyond PS
        policy_params, rollouts = ctrl.optimize(mjx_data, policy_params)
        obs = rollouts.observations[0, 0, 0]  # randomizations, rollouts, time

        # Apply the first control action
        u = ctrl.get_action(policy_params, 0.0)
        mjx_data = mjx.step(mjx_model, mjx_data.replace(ctrl=u))

        # Collect training data
        new_actions = policy_params.mean
        cost = jnp.sum(rollouts.costs[0, 0])  # First rollout is best for PS

        return (policy_params, mjx_data), TrainingData(
            obs=obs,
            old_action_sequence=old_actions,
            new_action_sequence=new_actions,
            cost=cost,
        )

    _, data = jax.lax.scan(
        _step, (policy_params, mjx_data), jnp.arange(num_steps)
    )
    return data


def visualize_dataset(data: TrainingData) -> None:
    """Visualize the collected training data on the mujoco viewer.

    Args:
        data: The training data, size (num_initial_conditions, num_steps, ...).
    """
    xml_path = ROOT + "/models/particle/scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    mj_data.mocap_pos = np.zeros((1, 3))

    num_initial_conditions = data.obs.shape[0]
    num_steps = data.obs.shape[1]

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        i = 0
        while viewer.is_running():
            print(f"  {i+1}/{num_initial_conditions}...", end="\r")
            for t in range(num_steps):
                st = time.time()
                mj_data.qpos[:] = data.obs[i, t, :2]
                mj_data.qvel[:] = data.obs[i, t, 2:]

                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                elapsed = time.time() - st
                if elapsed < mj_model.opt.timestep:
                    time.sleep(mj_model.opt.timestep - elapsed)

            i += 1
            if i == num_initial_conditions:
                i = 0


if __name__ == "__main__":
    # Set some parameters
    num_steps = 100
    num_initial_conditions = 128

    # Run data collection
    print("Collecting data...")
    st = time.time()
    rng = jax.random.key(0)
    rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.split(data_rng, num_initial_conditions)
    data = collect_data(num_steps, data_rng)
    collection_time = time.time() - st
    print(f"  Collection time: {collection_time:.2f} s")
    assert data.obs.shape == (num_initial_conditions, num_steps, 4)

    # Visualize the collected dataset
    print("Visualizing data...")
    visualize_dataset(data)

    # Save the dataset to a file
    print("Saving data...")
    with open("/tmp/gpc_particle_data.pkl", "wb") as f:
        pickle.dump(data, f)
