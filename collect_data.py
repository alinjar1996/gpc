from functools import partial
import jax
import jax.numpy as jnp
from mujoco import mjx
from hydrax.tasks.particle import Particle
from hydrax.algs import PredictiveSampling

"""
Collect training data by running the particle tracking task.
"""

if __name__ == "__main__":
    # Set up the task and control algorithm
    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.2)

    # Set up the optimizer
    policy_params = ctrl.init_params()
    jit_optimize = jax.jit(ctrl.optimize, donate_argnums=(1,))
    jit_get_action = jax.jit(ctrl.get_action)

    # Set up the simulator
    mjx_model = task.model
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=jnp.array([0.1, 0.2]))

    @partial(jax.jit, donate_argnums=(0,))
    def jit_step(data: mjx.Data, u: jax.Array) -> mjx.Data:
        """Step the dynamics with the given control input."""
        data = data.replace(ctrl=u)
        return mjx.step(mjx_model, data)

    # Run the simulation
    for t in range(100):
        policy_params, rollouts = jit_optimize(mjx_data, policy_params)
        u = jit_get_action(policy_params, 0.0)
        mjx_data = jit_step(mjx_data, u)

        print(mjx_data.qpos)
