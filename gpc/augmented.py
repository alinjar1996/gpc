from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, Trajectory


@dataclass
class PACParams:
    """Parameters for the policy-augmented controller.

    Attributes:
        base_params: The parameters for the base controller.
        policy_samples: Control sequences sampled from the policy.
    """

    base_params: Any
    policy_samples: jax.Array


class PolicyAugmentedController(SamplingBasedController):
    """An SPC generalization where samples are augmented by a learned policy."""

    def __init__(
        self,
        base_ctrl: SamplingBasedController,
        num_policy_samples: int,
    ) -> None:
        """Initialize the policy-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
            num_policy_samples: The number of samples to draw from the policy.
        """
        self.base_ctrl = base_ctrl
        self.num_policy_samples = num_policy_samples
        super().__init__(
            base_ctrl.task,
            base_ctrl.num_randomizations,
            base_ctrl.risk_strategy,
            seed=0,
        )

    def init_params(self) -> PACParams:
        """Initialize the controller parameters."""
        base_params = self.base_ctrl.init_params()
        policy_samples = jnp.zeros(
            (
                self.num_policy_samples,
                self.task.planning_horizon,
                self.task.model.nu,
            )
        )
        return PACParams(base_params=base_params, policy_samples=policy_samples)

    def sample_controls(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample control sequences from the base controller and the policy."""
        # Samples from the base controller
        base_samples, base_params = self.base_ctrl.sample_controls(
            params.base_params
        )

        # Include samples from the policy. Assumes that thes have already been
        # generated and stored in params.policy_samples.
        samples = jnp.append(base_samples, params.policy_samples, axis=0)

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
