from typing import Tuple
import jax
import jax.numpy as jnp
from mujoco import mjx
from hydrax.tasks.particle import Particle
from hydrax.algs import PredictiveSampling
from flax.struct import dataclass

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


def collect_data(ctrl: PredictiveSampling, num_steps) -> TrainingData:
    """Collect training data by running the particle tracking task."""
    # Set up the optimizer
    policy_params = ctrl.init_params()

    # Set up the simulator (which matches the controller's model exactly)
    mjx_model = ctrl.task.model
    mjx_data = mjx.make_data(mjx_model)

    # TODO: reset to a random initial state
    mjx_data = mjx_data.replace(qpos=jnp.array([0.1, 0.2]))

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

    _, data = jax.lax.scan(_step, (policy_params, mjx_data), jnp.arange(num_steps))
    return data


if __name__ == "__main__":
    # Set up the task and control algorithm
    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)

    # Run data collection
    data = collect_data(ctrl, num_steps=100)

    print(data.obs.shape)
    print(data.old_action_sequence.shape)
    print(data.new_action_sequence.shape)
    print(data.cost.shape)
