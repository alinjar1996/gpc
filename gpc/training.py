import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from hydrax.alg_base import SamplingBasedController

from gpc.augmented_controller import PredictionAugmentedController
from gpc.env import TrainingEnv


def train(
    env: TrainingEnv, ctrl: SamplingBasedController, visualize: bool
) -> None:
    """Train a generative predictive controller.

    Args:
        env: The training environment.
        ctrl: The controller to train.
        visualize: Flag for visualizing the training process.
    """
    if visualize:
        # Set up a visualization window
        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Set up the controller
    ctrl = PredictionAugmentedController(ctrl)
    psi = ctrl.init_params()

    # Initialize the environment
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    x = env.init_state(init_rng)

    # Training loop
    for t in range(env.episode_length):
        # Generate an action sequence from the learned policy
        y = env.get_observation(x)
        U_pred = jnp.zeros((ctrl.task.planning_horizon, ctrl.task.model.nu))

        # Using the predicted action sequence as one of the samples, find an
        # optimal action sequence
        psi = psi.replace(prediction=U_pred)
        psi, rollouts = ctrl.optimize(x.data, psi)
        U_star = ctrl.get_action_sequence(psi)

        # Record the cost of the predicted action sequence relative to the
        # best sample.
        costs = jnp.sum(rollouts.costs[0], axis=1)
        best_cost = jnp.min(costs)
        pred_cost = costs[-1]  # U_pred gets placed at the end of the samples

        # Step the simulation
        x = env.step(x, U_star[0])

        print(f"Step {t}: Best cost {best_cost}, Predicted cost {pred_cost}")

    if visualize:
        # Close the visualizer
        time.sleep(1)
        viewer.close()
