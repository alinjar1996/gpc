import jax
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.env import TrainingEnv


class ParticleEnv(TrainingEnv):
    """Training environment for the particle task."""

    def __init__(self, episode_length: int = 100) -> None:
        """Set up the particle training environment."""
        super().__init__(task=Particle(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, mocap_rng = jax.random.split(rng, 4)
        qpos = jax.random.uniform(pos_rng, (2,), minval=-0.29, maxval=0.29)
        qvel = jax.random.uniform(vel_rng, (2,), minval=-0.5, maxval=0.5)
        target = jax.random.uniform(mocap_rng, (2,), minval=-0.29, maxval=0.29)
        mocap_pos = data.mocap_pos.at[0, 0:2].set(target)
        return data.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)
