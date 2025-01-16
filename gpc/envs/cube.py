import jax
import jax.numpy as jnp
from hydrax.tasks.cube import CubeRotation
from mujoco import mjx

from gpc.envs import SimulatorState, TrainingEnv


class CubeEnv(TrainingEnv):
    """Training environment for the cube rotation task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the training environment."""
        super().__init__(task=CubeRotation(), episode_length=episode_length)
        self.q_home = jnp.array(self.task.mj_model.qpos0)

    def _random_quat(self, rng: jax.Array) -> jax.Array:
        """Generate a random quaternion."""
        u, v, w = jax.random.uniform(rng, (3,))
        return jnp.array(
            [
                jnp.sqrt(1 - u) * jnp.sin(2 * jnp.pi * v),
                jnp.sqrt(1 - u) * jnp.cos(2 * jnp.pi * v),
                jnp.sqrt(u) * jnp.sin(2 * jnp.pi * w),
                jnp.sqrt(u) * jnp.cos(2 * jnp.pi * w),
            ]
        )

    def episode_over(self, state: SimulatorState) -> bool:
        """Early termination if the cube is dropped."""
        episode_over = state.t >= self.episode_length
        cube_dropped = state.data.qpos[18] <= -0.2
        return jnp.logical_or(episode_over, cube_dropped)

    def goal_reached(self, state: SimulatorState) -> bool:
        """Check if the cube is close to the target orientation."""
        return (
            jnp.linalg.norm(self.task._get_cube_orientation_err(state.data))
            <= 0.1
        )

    def update_goal(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Update the target orientation."""
        target_quat = self._random_quat(rng)
        mocap_quat = data.mocap_quat.at[0].set(target_quat)
        return data.replace(mocap_quat=mocap_quat)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, joint_rng, cube_rng, target_rng = jax.random.split(rng, 4)

        # Resize configurations and velocities
        qpos = self.q_home + jax.random.uniform(
            joint_rng, self.task.model.nq, minval=-0.5, maxval=0.5
        )
        qvel = (
            jnp.zeros(self.task.model.nv)
            + jax.random.uniform(rng, self.task.model.nv) * 0.0
        )

        # Random quaternion for the cube
        cube_quat = self._random_quat(cube_rng)
        qpos = qpos.at[-4:].set(cube_quat)
        qpos = qpos.at[-7:-4].set(jnp.array([0.11, 0.0, 0.1]))

        # Random quaternion for the target
        target_quat = self._random_quat(target_rng)
        mocap_quat = data.mocap_quat.at[0].set(target_quat)

        # Reasonable spot for the target cube
        target_pos = jnp.array([[0.1, 0.2, 0.1]])
        mocap_pos = target_pos + jax.random.uniform(rng, (3,)) * 0.0

        return data.replace(
            qpos=qpos, qvel=qvel, mocap_quat=mocap_quat, mocap_pos=mocap_pos
        )

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the hand state and cube state relative to the target."""
        cube_position_err = self.task._get_cube_position_err(data)
        cube_orientation_err = self.task._get_cube_orientation_err(data)
        cube_orientation = data.qpos[-4:]
        cube_velocity = data.qvel[-6:]
        joint_pos = data.qpos[:16]
        joint_vel = data.qvel[:16]
        return jnp.concatenate(
            [
                cube_position_err,
                cube_orientation_err,
                cube_orientation,
                joint_pos,
                joint_vel,
                cube_velocity,
            ]
        )

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 48
