import sys

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.walker import Walker
from mujoco import mjx

from gpc.testing import test_interactive
from gpc.training import Policy
from gpc.utils import generate_dataset_and_save, train_policy_and_save


def reset_fn(mjx_data: mjx.Data, rng: jax.Array) -> mjx.Data:
    """Sample a random initial state."""
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)

    qpos = 0.01 * jax.random.normal(pos_rng, mjx_data.qpos.shape)
    qvel = 0.01 * jax.random.normal(vel_rng, mjx_data.qvel.shape)

    return mjx_data.replace(qpos=qpos, qvel=qvel)


if __name__ == "__main__":
    # Set parameters
    task = Walker()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.5)
    num_timesteps = 1000
    num_resets = 128
    hidden_layers = [128, 128]

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
            fname="/tmp/gpc_walker_data.pkl",
        )
    if fit:
        train_policy_and_save(
            task,
            dataset_fname="/tmp/gpc_walker_data.pkl",
            policy_fname="/tmp/gpc_walker_policy.pkl",
            hidden_layers=hidden_layers,
            epochs=5000,
            print_every=50,
            batch_size=1024,
            learning_rate=3e-4,
        )
    if deploy:
        policy = Policy.load("/tmp/gpc_walker_policy.pkl")
        test_interactive(task, policy)
