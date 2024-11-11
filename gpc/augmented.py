from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, Trajectory


@dataclass
class PACParams:
    """Parameters for the prediction-augmented controller.

    Attributes:
        base_params: The policy parameters for the base controller.
        prediction: The predicted control sequence to augment the samples.
    """

    base_params: Any
    prediction: jax.Array


class PredictionAugmentedController(SamplingBasedController):
    """An SPC generalization where one sample is replaced by a prediction.

    This prediction could come from a learned model or a heuristic, and help
    guide the planner towards more informative samples.
    """

    def __init__(self, base_ctrl: SamplingBasedController) -> None:
        """Initialize the prediction-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
        """
        self.base_ctrl = base_ctrl
        super().__init__(
            base_ctrl.task,
            base_ctrl.num_randomizations,
            base_ctrl.risk_strategy,
            seed=0,
        )

    def init_params(self) -> PACParams:
        """Initialize the controller parameters."""
        base_params = self.base_ctrl.init_params()
        prediction = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        return PACParams(base_params=base_params, prediction=prediction)

    def sample_controls(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample control sequences, overridding one with the prediciton."""
        samples, base_params = self.base_ctrl.sample_controls(
            params.base_params
        )
        samples = samples.at[-1].set(params.prediction)
        return samples, params.replace(base_params=base_params)

    def update_params(
        self, params: PACParams, rollouts: Trajectory
    ) -> PACParams:
        """Update the policy parameters according to the base controller."""
        base_params = self.base_ctrl.update_params(params.base_params, rollouts)
        return params.replace(base_params=base_params)

    def get_action(self, params: PACParams, t: float) -> jax.Array:
        """Get the action from the base controller at a given time."""
        return self.base_ctrl.get_action(params.base_params, t)

    def get_action_sequence(self, params: PACParams) -> jax.Array:
        """Get the action sequence from the controller."""
        timesteps = jnp.arange(self.task.planning_horizon) * self.task.dt
        return jax.vmap(self.get_action, in_axes=(None, 0))(params, timesteps)
