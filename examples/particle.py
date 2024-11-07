import time

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.architectures import ScoreMLP
from gpc.dataset import TrainingData, collect_data, visualize_data
from gpc.testing import test_interactive
from gpc.training import Policy, train


def reset_fn(mjx_data: mjx.Data, rng: jax.Array) -> mjx.Data:
    """Sample a random initial state for the particle."""
    rng, pos_rng, vel_rng, mocap_rng = jax.random.split(rng, 4)
    qpos = jax.random.uniform(pos_rng, (2,), minval=-0.29, maxval=0.29)
    qvel = jax.random.uniform(vel_rng, (2,), minval=-0.5, maxval=0.5)
    target = jax.random.uniform(mocap_rng, (2,), minval=-0.29, maxval=0.29)
    mocap_pos = mjx_data.mocap_pos.at[0, 0:2].set(target)
    return mjx_data.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)


def gather_dataset(
    fname: str = "/tmp/gpc_particle_data.pkl", visualize: bool = True
) -> None:
    """Gather a dataset for training the GPC policy."""
    print("Collecting data...")
    rng = jax.random.key(0)
    num_timesteps = 100
    num_resets = 32

    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.5)

    st = time.time()
    rng, reset_rng = jax.random.split(rng)
    dataset = collect_data(
        task, ctrl, num_timesteps, num_resets, reset_fn, reset_rng
    )
    elapsed = time.time() - st

    N = num_resets * num_timesteps
    print(f"  Collected {N} data points in {elapsed:.2f}s")

    dataset.save(fname)
    print(f"  Dataset saved to {fname}")

    if visualize:
        visualize_data(task, dataset)


def train_policy(
    dataset_fname: str = "/tmp/gpc_particle_data.pkl",
    policy_fname: str = "/tmp/gpc_particle_policy.pkl",
) -> None:
    """Train a GPC policy and save it to a file."""
    print("Training policy...")
    task = Particle()
    dataset = TrainingData.load(dataset_fname)
    net = ScoreMLP([64, 64])

    policy = train(dataset, task, net)

    policy.save(policy_fname)
    print(f"  Policy saved to {policy_fname}")


def test(policy_fname: str = "/tmp/gpc_particle_policy.pkl") -> None:
    """Test the trained policy interactively."""
    print("Testing policy...")
    task = Particle()
    policy = Policy.load(policy_fname)
    test_interactive(task, policy)


if __name__ == "__main__":
    # Run predictive sampling and save out the dataset.
    # gather_dataset(visualize=True)

    # Train a GPC policy on the dataset and save the policy.
    train_policy()

    # Load the saved policy and test with an interactive simulation.
    test()
