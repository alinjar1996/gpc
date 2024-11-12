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


class PolicyAugmentedController(SamplingBasedController):
    """An SPC generalization where samples are augmented by a learned policy."""

    def __init__(
        self,
        base_ctrl: SamplingBasedController,
        num_policy_samples: int,
        policy_noise_level: int,
    ) -> None:
        """Initialize the policy-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
            num_policy_samples: The number of samples to draw from the policy
                                distribution. For now we assume this is Gaussian
                                with a fixed variance.
            policy_noise_level: The standard deviation of iid Gaussian noise to
                                add to the policy samples.
        """
        self.base_ctrl = base_ctrl
        super().__init__(
            base_ctrl.task,
            base_ctrl.num_randomizations,
            base_ctrl.risk_strategy,
            seed=0,
        )
        self.num_policy_samples = num_policy_samples
        self.policy_noise_level = policy_noise_level

    def init_params(self) -> PACParams:
        """Initialize the controller parameters."""
        base_params = self.base_ctrl.init_params()
        prediction = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        return PACParams(base_params=base_params, prediction=prediction)

    def sample_controls(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample control sequences from the base controller and the policy."""
        # Samples from the base controller
        samples, base_params = self.base_ctrl.sample_controls(
            params.base_params
        )

        # Samples from the policy
        rng = base_params.rng
        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(
            noise_rng,
            (
                self.num_policy_samples,
                self.task.planning_horizon,
                self.task.model.nu,
            ),
        )
        pred_samples = params.prediction + self.policy_noise_level * noise
        samples = jnp.append(pred_samples, samples, axis=0)
        base_params = base_params.replace(rng=rng)

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
