import jax
import jax.numpy as jnp
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.augmented_controller import PredictionAugmentedController


def test_augmented() -> None:
    """Test the prediction-augmented controller."""
    # Task and optimizer setup
    task = Particle()
    ps = PredictiveSampling(task, num_samples=32, noise_level=0.1)
    opt = PredictionAugmentedController(ps)
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    state = state.replace(
        mocap_pos=state.mocap_pos.at[0, 0:2].set(jnp.array([0.01, 0.01]))
    )
    params = opt.init_params()
    params = params.replace(
        prediction=jnp.ones((task.planning_horizon, task.model.nu))
    )

    for _ in range(10):
        # Do an optimization step
        params, rollouts = jit_opt(state, params)

    # Pick the best rollout (first axis is for domain randomization, unused)
    total_costs = jnp.sum(rollouts.costs[0], axis=1)
    best_idx = jnp.argmin(total_costs)
    best_obs = rollouts.observations[0, best_idx]
    best_ctrl = rollouts.controls[0, best_idx]

    assert jnp.linalg.norm(best_obs[-1, 0:2]) < 0.01
    assert jnp.all(best_ctrl != 0.0)
    assert jnp.all(params.prediction == 1.0)

    U = opt.get_action_sequence(params)
    assert jnp.allclose(U, params.base_params.mean)


if __name__ == "__main__":
    test_augmented()
