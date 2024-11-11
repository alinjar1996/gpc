import time
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from hydrax.alg_base import SamplingBasedController

from gpc.augmented_controller import PACParams, PredictionAugmentedController
from gpc.env import SimulatorState, TrainingEnv


def train(env: TrainingEnv, ctrl: SamplingBasedController) -> None:
    """Train a generative predictive controller.

    Args:
        env: The training environment.
        ctrl: The controller to train.
    """
    ctrl = PredictionAugmentedController(ctrl)

    def _scan_fn(carry: Tuple[SimulatorState, PACParams], t: int) -> Tuple:
        """Take step in the training loop."""
        x, psi = carry

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

        return (x, psi), (y, U_star, best_cost, pred_cost)

    @jax.jit
    def _run_episode(
        rng: jax.Array, params: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Simulate an episode and collect policy training data.

        Args:
            rng: The random number generator key.
            params: The policy network parameters.

        Returns:
            Observations y.
            Optimal actions U.
            Average cost of the best sample.
            Average cost of the policy's sample.
        """
        rng, ctrl_rng, env_rng = jax.random.split(rng, 3)

        # Set the initial state of the environment
        x = env.init_state(env_rng)

        # Set the initial sampling-based controller parameters
        psi = ctrl.init_params()
        psi = psi.replace(base_params=psi.base_params.replace(rng=ctrl_rng))

        _, (y, U, best_costs, pred_costs) = jax.lax.scan(
            _scan_fn, (x, psi), jnp.arange(env.episode_length)
        )

        return y, U, jnp.mean(best_costs), jnp.mean(pred_costs)

    rng = jax.random.key(0)
    rng, episode_rng = jax.random.split(rng)
    episode_rngs = jax.random.split(episode_rng, 10)

    st = time.time()
    y, U, best, pred = jax.vmap(_run_episode, in_axes=(0, None))(
        episode_rngs, None
    )
    print(y.shape, U.shape, best, pred)
    print("Time taken:", time.time() - st)
