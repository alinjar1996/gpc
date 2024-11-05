from pathlib import Path

import jax
import jax.numpy as jnp
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.data import TrainingData, collect_data, visualize_data


def test_collect_data() -> None:
    """Test basic data collection."""
    rng = jax.random.key(0)
    num_timesteps = 12
    num_resets = 4

    task = Particle()
    ctrl = PredictiveSampling(task, num_samples=8, noise_level=0.5)

    def _reset_fn(mjx_data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Sample a random initial state for the particle."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        qpos = jax.random.uniform(pos_rng, (2,), minval=-0.29, maxval=0.29)
        qvel = jax.random.uniform(vel_rng, (2,), minval=-0.5, maxval=0.5)
        return mjx_data.replace(qpos=qpos, qvel=qvel)

    rng, reset_rng = jax.random.split(rng)
    dataset = collect_data(
        task, ctrl, num_timesteps, num_resets, _reset_fn, reset_rng
    )

    assert isinstance(dataset, TrainingData)
    assert dataset.observation.shape == (num_resets, num_timesteps, 4)
    assert dataset.old_action_sequence.shape == (
        num_resets,
        num_timesteps,
        5,
        2,
    )
    assert dataset.new_action_sequence.shape == (
        num_resets,
        num_timesteps,
        5,
        2,
    )
    assert dataset.state.qpos.shape == (num_resets, num_timesteps, 2)

    if __name__ == "__main__":
        # Only visualize the data if running this script directly
        visualize_data(task, dataset)


def test_save_data() -> None:
    """Test saving a dataset to disc."""
    # Make some fake data
    dataset = TrainingData(
        observation=jnp.zeros((4, 5)),
        old_action_sequence=jnp.zeros((5, 2)),
        new_action_sequence=jnp.zeros((5, 2)),
        state=jnp.zeros((2,)),
    )

    # Create a temporary file to save the data
    local_dir = Path("_test_save_data")
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / "test_data.pkl"

    # Save the data to the file
    dataset.save(path)
    assert path.is_file()

    # Clean up the temporary file
    for file in local_dir.iterdir():
        file.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_collect_data()
