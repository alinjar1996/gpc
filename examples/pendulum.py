import time

import jax
from hydrax.algs import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum
from mujoco import mjx

from gpc.architectures import ScoreMLP
from gpc.dataset import TrainingData, collect_data, visualize_data
from gpc.testing import test_interactive
from gpc.training import Policy, train


def reset_fn(mjx_data: mjx.Data, rng: jax.Array) -> mjx.Data:
    """Sample a random initial state for the pendulum."""
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    qpos = jax.random.uniform(pos_rng, (1,), minval=-3.14, maxval=3.14)
    qvel = jax.random.uniform(vel_rng, (1,), minval=-3.0, maxval=3.0)
    return mjx_data.replace(qpos=qpos, qvel=qvel)


def gather_dataset(
    fname: str = "/tmp/gpc_pendulum_data.pkl", visualize: bool = True
) -> None:
    """Gather a dataset for training the GPC policy."""
    print("Collecting data...")
    rng = jax.random.key(0)
    num_timesteps = 400
    num_resets = 128

    task = Pendulum()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.1)

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
    dataset_fname: str = "/tmp/gpc_pendulum_data.pkl",
    policy_fname: str = "/tmp/gpc_pendulum_policy.pkl",
) -> None:
    """Train a GPC policy and save it to a file."""
    print("Training policy...")
    task = Pendulum()
    dataset = TrainingData.load(dataset_fname)
    net = ScoreMLP([64, 64])
    policy = train(dataset, task, net)
    policy.save(policy_fname)
    print(f"  Policy saved to {policy_fname}")


def test(policy_fname: str = "/tmp/gpc_pendulum_policy.pkl") -> None:
    """Test the GPC policy interactively."""
    print("Testing policy...")
    policy = Policy.load(policy_fname)
    task = Pendulum()
    test_interactive(task, policy)


if __name__ == "__main__":
    # gather_dataset()
    train_policy()
    test()
