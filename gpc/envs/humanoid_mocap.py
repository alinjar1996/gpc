import jax
import jax.numpy as jnp
from hydrax.tasks.humanoid_mocap import HumanoidMocap
from mujoco import mjx

from gpc.envs import TrainingEnv


class HumanoidMocapEnv(TrainingEnv):
    """Training environment for humanoid (Unitree G1) standup."""

    def __init__(
        self,
        episode_length: int,
        reference_filename: str = "walk1_subject1.csv",
    ) -> None:
        """Set up the humanoid mocap tracking training environment.

        Args:
            episode_length: number of time steps in each episode.
            reference_filename: CSV file holding the mocap reference.

        Available reference files can be found at
        https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset/tree/main/g1
        """
        super().__init__(task=HumanoidMocap(), episode_length=episode_length)

        # Length of the reference trajectory (which is at 30 Hz)
        # self.t_max = task.reference.shape[0] / 30.0
        self.t_max = 5.0  # DEBUG: set a shorter sequence

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, time_rng = jax.random.split(rng, 4)

        # Random start times
        t = jax.random.uniform(time_rng, minval=0.0, maxvel=self.t_max)

        # Mocap reference at this time
        q_ref = self.task._get_reference_configuration(t)

        # Random positions and velocities, centered on the reference
        qpos = q_ref + 0.01 * jax.random.normal(pos_rng, (self.task.model.nq,))
        qvel = 0.01 * jax.random.normal(vel_rng, (self.task.model.nv,))

        return data.replace(time=t, qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the full state and target mocap position."""
        # TODO(vincekurtz): consider using (or including) the reference a few
        # steps ahead.
        q_ref = self.task._get_reference_configuration(data.time)
        return jnp.concatenate([q_ref, data.qpos, data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observations."""
        return 68  # TODO: fix
