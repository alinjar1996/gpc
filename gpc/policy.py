import pickle
from pathlib import Path
from typing import Any, Union

import jax
import jax.numpy as jnp

from gpc.architectures import DenoisingMLP


class Policy:
    """A pickle-able Generative Predictive Control policy."""

    def __init__(
        self,
        net: DenoisingMLP,
        params: Any,
        u_min: jax.Array,
        u_max: jax.Array,
        dt: float = 0.1,
    ):
        """Create a new GPC policy, which generates actions with flow matching.

        Args:
            net: The flow matching network that generates the action sequence.
            params: The parameters of the network.
            u_min: The minimum action values.
            u_max: The maximum action values.
            dt: The integration step size for flow matching.
        """
        self.net = net
        self.params = params
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt

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

    def apply(self, prev: jax.Array, y: jax.Array, rng: jax.Array) -> jax.Array:
        """Generate an action sequence conditioned on the observation.

        Args:
            prev: The previous action sequence.
            y: The current observation.
            rng: The random number generator key.

        Returns:
            The updated action sequence
        """
        # TODO: consider warm-starting from prev somehow
        U = jax.random.normal(rng, prev.shape)
        for t in jnp.arange(0.0, 1.0, self.dt):
            v = self.net.apply(self.params, U, y, jnp.array([t]))
            U += self.dt * v
            U = jax.numpy.clip(U, self.u_min, self.u_max)

        return U
