from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
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


if __name__ == "__main__":
    # Set some parameters
    num_steps = 100
    num_initial_conditions = 10

    # Run data collection
    rng = jax.random.key(0)
    rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.split(data_rng, num_initial_conditions)
    data = collect_data(num_steps, data_rng)

    print(data.obs.shape)
    print(data.old_action_sequence.shape)
    print(data.new_action_sequence.shape)
    print(data.cost.shape)
