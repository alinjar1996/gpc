import pickle
from pathlib import Path
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass

from gpc.architectures import DenoisingMLP


@dataclass
class Policy:
    """A pickle-able Generative Predictive Control policy.

    Generates action sequences using flow matching, conditioned on the latest
    observation, e.g., samples U = [u_0, u_1, ...] ~ p(U | y).

    Attributes:
        model: The flow matching network that generates the action sequence.
        u_min: The minimum action values.
        u_max: The maximum action values.
        dt: The integration step size for flow matching.
    """

    model: DenoisingMLP
    u_min: jax.Array
    u_max: jax.Array
    dt: float = 0.1

    def save(self, path: Union[Path, str]) -> None:
        """Save the policy to a file.

        Args:
            path: The path to save the policy to.
        """
        model_args = {
            "action_size": self.model.action_size,
            "observation_size": self.model.observation_size,
            "horizon": self.model.horizon,
            "hidden_layers": self.model.hidden_layers,
        }
        policy_args = {
            "u_min": self.u_min,
            "u_max": self.u_max,
            "dt": self.dt,
        }
        _, state = nnx.split(self.model)
        data = {
            "model_args": model_args,
            "policy_args": policy_args,
            "state": state,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "Policy":
        """Load a policy from a file.

        Args:
            path: The path to load the policy from.

        Returns:
            The loaded policy instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        empty_model = DenoisingMLP(**data["model_args"], rngs=nnx.Rngs(0))
        graphdef, _ = nnx.split(empty_model)
        model = nnx.merge(graphdef, data["state"])

        return Policy(model, **data["policy_args"])

    def apply(
        self,
        prev: jax.Array,
        y: jax.Array,
        rng: jax.Array,
        warm_start_level: float = 0.0,
    ) -> jax.Array:
        """Generate an action sequence conditioned on the observation.

        Args:
            prev: The previous action sequence.
            y: The current observation.
            rng: The random number generator key.
            warm_start_level: The degree of warm-starting to use, in [0, 1].

        A warm-start level of 0.0 means the action sequence is generated from
        scratch, with the seed for flow matching drawn from a random normal
        distribution. A warm-start level of 1.0 means the seed is the previous
        action sequence. Values in between interpolate between these two, with
        larger values giving smoother but less exploratory action sequences.

        Returns:
            The updated action sequence
        """
        # Set the initial sample
        warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        noise = jax.random.normal(rng, prev.shape)
        U = warm_start_level * prev + (1 - warm_start_level) * noise

        def _step(args: Tuple[jax.Array, float]) -> Tuple[jax.Array, float]:
            """Flow the sample U along the learned vector field."""
            U, t = args
            U += self.dt * self.model(U, y, t)
            U = jax.numpy.clip(U, self.u_min, self.u_max)
            return U, t + self.dt

        # While t < 1, U += dt * model(U, y, t)
        U, t = jax.lax.while_loop(
            lambda args: jnp.all(args[1] < 1.0),
            _step,
            (U, jnp.zeros(1)),
        )
        return U
