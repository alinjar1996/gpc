import time
from typing import Callable, Sequence

import jax
from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task
from mujoco import mjx

from gpc.architectures import ActionSequenceMLP
from gpc.dataset import TrainingData, collect_data, visualize_data
from gpc.training import train

"""
Various helper functions.
"""


def generate_dataset_and_save(
    task: Task,
    ctrl: SamplingBasedController,
    reset_fn: Callable[[mjx.Data, jax.Array], mjx.Data],
    num_timesteps: int,
    num_resets: int,
    fname: str,
    seed: int = 0,
    visualize: bool = True,
) -> None:
    """Generate a training dataset and save it to a file."""
    print("Collecting data...")
    rng = jax.random.PRNGKey(seed)

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


def train_policy_and_save(
    task: Task,
    dataset_fname: str,
    policy_fname: str,
    hidden_layers: Sequence[int],
) -> None:
    """Load a dataset, train a policy, and save the policy to a file."""
    dataset = TrainingData.load(dataset_fname)
    net = ActionSequenceMLP(hidden_layers, task.planning_horizon, task.model.nu)
    policy = train(dataset, task, net)
    policy.save(policy_fname)
    print(f"  Policy saved to {policy_fname}")
