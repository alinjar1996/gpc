import sys

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum
from mujoco import mjx

from gpc.testing import test_interactive
from gpc.training import Policy
from gpc.utils import generate_dataset_and_save, train_policy_and_save


def reset_fn(mjx_data: mjx.Data, rng: jax.Array) -> mjx.Data:
    """Sample a random initial state for the pendulum."""
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    qpos = jax.random.uniform(pos_rng, (1,), minval=-3.14, maxval=3.14)
    qvel = jax.random.uniform(vel_rng, (1,), minval=-3.0, maxval=3.0)
    return mjx_data.replace(qpos=qpos, qvel=qvel)


if __name__ == "__main__":
    # Set parameters
    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.1)
    num_timesteps = 100
    num_resets = 128
    hidden_layers = [64, 64]

    # Choose what to do based on command-line arguments
    generate, fit, deploy = True, True, True
    if len(sys.argv) == 2:
        generate = sys.argv[1] == "generate"
        fit = sys.argv[1] == "fit"
        deploy = sys.argv[1] == "deploy"

    if generate:
        generate_dataset_and_save(
            task,
            ctrl,
            reset_fn,
            num_timesteps,
            num_resets,
            fname="/tmp/gpc_pendulum_data.pkl",
        )
    if fit:
        train_policy_and_save(
            task,
            dataset_fname="/tmp/gpc_pendulum_data.pkl",
            policy_fname="/tmp/gpc_pendulum_policy.pkl",
            hidden_layers=hidden_layers,
        )
    if deploy:
        policy = Policy.load("/tmp/gpc_pendulum_policy.pkl")
        test_interactive(task, policy)
