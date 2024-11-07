import sys

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.testing import test_interactive
from gpc.training import Policy
from gpc.utils import generate_dataset_and_save, train_policy_and_save


def reset_fn(mjx_data: mjx.Data, rng: jax.Array) -> mjx.Data:
    """Sample a random initial state for the particle."""
    rng, pos_rng, vel_rng, mocap_rng = jax.random.split(rng, 4)
    qpos = jax.random.uniform(pos_rng, (2,), minval=-0.29, maxval=0.29)
    qvel = jax.random.uniform(vel_rng, (2,), minval=-0.5, maxval=0.5)
    target = jax.random.uniform(mocap_rng, (2,), minval=-0.29, maxval=0.29)
    mocap_pos = mjx_data.mocap_pos.at[0, 0:2].set(target)
    return mjx_data.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)


if __name__ == "__main__":
    # Set parameters
    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.1)
    num_timesteps = 100
    num_resets = 32
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
            fname="/tmp/gpc_particle_data.pkl",
        )
    if fit:
        train_policy_and_save(
            task,
            dataset_fname="/tmp/gpc_particle_data.pkl",
            policy_fname="/tmp/gpc_particle_policy.pkl",
            hidden_layers=hidden_layers,
        )
    if deploy:
        policy = Policy.load("/tmp/gpc_particle_policy.pkl")
        test_interactive(task, policy)
