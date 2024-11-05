import pickle
from pathlib import Path
from typing import Any, Union

import flax.linen as nn
import jax

Params = Any


class Policy:
    """A pickle-able Generative Predictive Control policy."""

    def __init__(
        self, net: nn.Module, params: Any, u_min: jax.Array, u_max: jax.Array
    ):
        """Create a new GPC policy.

        Args:
            net: The network that maps action sequence and observation to a new
                 action sequence.
            params: The parameters of the network.
            u_min: The minimum action values.
            u_max: The maximum action values.
        """
        self.net = net
        self.params = params
        self.u_min = u_min
        self.u_max = u_max

    def save(self, path: Union[Path, str]) -> None:
        """Save the policy to a file.

        Args:
            path: The path to save the policy to.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "Policy":
        """Load a policy from a file.

        Args:
            path: The path to load the policy from.

        Returns:
            The loaded policy instance
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def apply(self, u: jax.Array, y: jax.Array) -> jax.Array:
        """Apply the policy, updating the action sequence.

        Args:
            u: The current action sequence.
            y: The current observation.

        Returns:
            The updated action sequence
        """
        u_new = self.net.apply(self.params, u, y)
        return jax.numpy.clip(u_new, self.u_min, self.u_max)
