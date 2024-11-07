import time

import jax
import jax.numpy as jnp
from hydrax.algs import PredictiveSampling
from hydrax.tasks.cart_pole import CartPole
from mujoco import mjx

from gpc.architectures import ActionSequenceMLP
from gpc.dataset import TrainingData, collect_data, visualize_data
from gpc.testing import test_interactive
from gpc.training import Policy, train


def reset_fn(mjx_data: mjx.Data, rng: jax.Array) -> mjx.Data:
    """Sample a random initial state."""
    rng, theta_rng, pos_rng, vel_rng = jax.random.split(rng, 4)

    theta = jax.random.uniform(theta_rng, (), minval=-3.14, maxval=3.14)
    pos = jax.random.uniform(pos_rng, (), minval=-1.8, maxval=1.8)
    qvel = jax.random.uniform(vel_rng, (2,), minval=-2.0, maxval=2.0)
    qpos = jnp.array([pos, theta])

    return mjx_data.replace(qpos=qpos, qvel=qvel)


def gather_dataset(
    fname: str = "/tmp/gpc_cart_pole_data.pkl", visualize: bool = True
) -> None:
    """Gather a dataset for training the GPC policy."""
    print("Collecting data...")
    rng = jax.random.key(0)
    num_timesteps = 200
    num_resets = 128

    task = CartPole()
    ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.3)

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
    dataset_fname: str = "/tmp/gpc_cart_pole_data.pkl",
    policy_fname: str = "/tmp/gpc_cart_pole_policy.pkl",
) -> None:
    """Train a GPC policy and save it to a file."""
    print("Training policy...")
    task = CartPole()
    dataset = TrainingData.load(dataset_fname)
    net = ActionSequenceMLP([64, 64], task.planning_horizon, task.model.nu)
    policy = train(dataset, task, net)
    policy.save(policy_fname)
    print(f"  Policy saved to {policy_fname}")


def test(policy_fname: str = "/tmp/gpc_cart_pole_policy.pkl") -> None:
    """Test the GPC policy interactively."""
    print("Testing policy...")
    policy = Policy.load(policy_fname)
    task = CartPole()
    test_interactive(task, policy)


if __name__ == "__main__":
    gather_dataset()
    train_policy()
    test()
