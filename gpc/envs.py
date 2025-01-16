from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax.struct import dataclass
from hydrax.task_base import Task
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.crane import Crane
from hydrax.tasks.cube import CubeRotation
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.tasks.humanoid import Humanoid
from hydrax.tasks.particle import Particle
from hydrax.tasks.pendulum import Pendulum
from hydrax.tasks.pusht import PushT
from hydrax.tasks.walker import Walker
from mujoco import mjx


@dataclass
class SimulatorState:
    """A dataclass for storing the simulator state.

    Attributes:
        data: The mjx simulator data.
        t: The current time step.
        rng: The random number generator key.
    """

    data: mjx.Data
    t: int
    rng: jax.Array


class TrainingEnv(ABC):
    """Abstract class defining a training environment."""

    def __init__(self, task: Task, episode_length: int) -> None:
        """Initialize the training environment."""
        self.task = task
        self.episode_length = episode_length
        self.renderer = mujoco.Renderer(self.task.mj_model)

        # Disable shadows and reflections for faster rendering
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False

    def init_state(self, rng: jax.Array) -> SimulatorState:
        """Initialize the simulator state."""
        state = SimulatorState(
            data=mjx.make_data(self.task.model), t=0, rng=rng
        )
        return self._reset_state(state)

    def render(self, states: SimulatorState, fps: int = 10) -> np.ndarray:
        """Render video frames from a state trajectory.

        Note that this is not a pure jax function, and should only be used for
        visualization.

        Args:
            states: Sequence of states (vmapped over time).
            fps: The frames per second for the video.

        Returns:
            A sequence of video frames, with shape (T, C, H, W).
        """
        sim_dt = self.task.model.opt.timestep
        render_dt = 1.0 / fps
        render_every = int(round(render_dt / sim_dt))
        steps = np.arange(0, len(states.t), render_every)

        frames = []
        for i in steps:
            mjx_data = jax.tree.map(lambda x: x[i], states.data)  # noqa: B023
            mj_data = mjx.get_data(self.task.mj_model, mjx_data)
            self.renderer.update_scene(mj_data)
            pixels = self.renderer.render()  # H, W, C
            frames.append(pixels.transpose(2, 0, 1))  # C, H, W

        return np.stack(frames)

    @abstractmethod
    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""

    @abstractmethod
    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Get the observation from the simulator state."""

    @property
    @abstractmethod
    def observation_size(self) -> int:
        """The size of the observation space."""

    def _reset_state(self, state: SimulatorState) -> SimulatorState:
        """Reset the simulator state to start a new episode."""
        rng, reset_rng = jax.random.split(state.rng)
        data = self.reset(state.data, reset_rng)
        data = mjx.forward(self.task.model, data)  # update sensor data
        return SimulatorState(data=data, t=0, rng=rng)

    def _update_goal(self, state: SimulatorState) -> SimulatorState:
        """Update the goal state during the middle of an episode."""
        rng, goal_rng = jax.random.split(state.rng)
        data = self.update_goal(state.data, goal_rng)
        return state.replace(data=data, rng=rng)

    def _get_observation(self, state: SimulatorState) -> jax.Array:
        """Get the observation from the simulator state."""
        return self.get_obs(state.data)

    def episode_over(self, state: SimulatorState) -> bool:
        """Check if the episode is over.

        Override this method if the episode should terminate early.
        """
        return state.t >= self.episode_length

    def goal_reached(self, state: SimulatorState) -> bool:
        """Check if we've achieved a sub-goal.

        This gives us the opportunity to update the goal before the episode
        ends. For example, we might want to choose a new target configuration
        once the old one has been reached.
        """
        return False

    def update_goal(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Update the goal state during the middle of an episode.

        Typically this is done via mocap_pos and mocap_quat, and by default we
        do nothing.
        """
        return data

    def step(self, state: SimulatorState, action: jax.Array) -> SimulatorState:
        """Take a simulation step.

        Args:
            state: The simulator state.
            action: The action to take.

        Returns:
            The new simulator state and the new time step.
        """
        # Check if the episode is over
        next_state = jax.lax.cond(
            self.episode_over(state),
            lambda _: self._reset_state(state),
            lambda _: state.replace(
                data=mjx.step(self.task.model, state.data.replace(ctrl=action)),
                t=state.t + 1,
            ),
            operand=None,
        )

        # Check if we've reached a sub-goal that needs updating
        next_state = jax.lax.cond(
            self.goal_reached(next_state),
            lambda _: self._update_goal(next_state),
            lambda _: next_state,
            operand=None,
        )

        return next_state


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

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the position relative to the target and the velocity."""
        pos = (
            data.site_xpos[self.task.pointmass_id, 0:2] - data.mocap_pos[0, 0:2]
        )
        vel = data.qvel[:]
        return jnp.concatenate([pos, vel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 4


class PendulumEnv(TrainingEnv):
    """Training environment for the pendulum swingup task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the pendulum training environment."""
        super().__init__(
            task=Pendulum(planning_horizon=5),
            episode_length=episode_length,
        )

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        qpos = jax.random.uniform(pos_rng, (1,), minval=-jnp.pi, maxval=jnp.pi)
        qvel = jax.random.uniform(vel_rng, (1,), minval=-8.0, maxval=8.0)
        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the velocity and sin/cos of the angle."""
        theta = data.qpos[0]
        theta_dot = data.qvel[0]
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])

    @property
    def observation_size(self) -> int:
        """The size of the observation space (sin, cos, theta_dot)."""
        return 3


class CartPoleEnv(TrainingEnv):
    """Training environment for the cartpole swingup task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the cartpole training environment."""
        super().__init__(task=CartPole(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, theta_rng, pos_rng, vel_rng = jax.random.split(rng, 4)

        theta = jax.random.uniform(theta_rng, (), minval=-3.14, maxval=3.14)
        pos = jax.random.uniform(pos_rng, (), minval=-1.8, maxval=1.8)
        qvel = jax.random.uniform(vel_rng, (2,), minval=-2.0, maxval=2.0)
        qpos = jnp.array([pos, theta])

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the velocity and sin/cos of the angle."""
        p = data.qpos[0]
        theta = data.qpos[1]
        v = data.qvel[0]
        theta_dot = data.qvel[1]
        return jnp.array([p, jnp.cos(theta), jnp.sin(theta), v, theta_dot])

    @property
    def observation_size(self) -> int:
        """The size of the observation space (includes sin and cos)."""
        return 5


class DoubleCartPoleEnv(TrainingEnv):
    """Training environment for the double cart-pole swingup task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the training environment."""
        super().__init__(task=DoubleCartPole(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, theta_rng, pos_rng, vel_rng = jax.random.split(rng, 4)

        thetas = jax.random.uniform(theta_rng, (2), minval=-3.14, maxval=3.14)
        pos = jax.random.uniform(pos_rng, (), minval=-2.8, maxval=2.8)
        qvel = jax.random.uniform(vel_rng, (3,), minval=-10.0, maxval=10.0)
        qpos = jnp.array([pos, thetas[0], thetas[1]])

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the velocity and sin/cos of the angles."""
        p = data.qpos[0]
        theta1 = data.qpos[1]
        theta2 = data.qpos[2]
        q_obs = jnp.array(
            [
                p,
                jnp.cos(theta1),
                jnp.sin(theta1),
                jnp.cos(theta2),
                jnp.sin(theta2),
            ]
        )
        return jnp.concatenate([q_obs, data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space (includes sin and cos)."""
        return 8


class WalkerEnv(TrainingEnv):
    """Training environment for the walker task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the walker training environment."""
        super().__init__(task=Walker(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)

        # Joint limits are zero for the floating base
        q_min = self.task.model.jnt_range[:, 0]
        q_max = self.task.model.jnt_range[:, 1]
        q_min = q_min.at[2].set(-1.5)  # orientation
        q_max = q_max.at[2].set(1.5)
        qpos = jax.random.uniform(pos_rng, (9,), minval=q_min, maxval=q_max)
        qvel = jax.random.uniform(vel_rng, (9,), minval=-0.1, maxval=0.1)

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe everything in the state except the horizontal position."""
        pz = data.qpos[0]  # base coordinates are (z, x, theta)
        theta = data.qpos[2]
        base_pos_data = jnp.array([jnp.cos(theta), jnp.sin(theta), pz])
        return jnp.concatenate([base_pos_data, data.qpos[3:], data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 18


class PushTEnv(TrainingEnv):
    """Training environment for the pusher-T task."""

    def __init__(self, episode_length: int) -> None:
        """Set up the walker training environment."""
        super().__init__(task=PushT(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, goal_pos_rng, goal_ori_rng = jax.random.split(
            rng, 5
        )

        # Random configuration for the pusher and the T
        q_min = jnp.array([-0.2, -0.2, -jnp.pi, -0.2, -0.2])
        q_max = jnp.array([0.2, 0.2, jnp.pi, 0.2, 0.2])
        qpos = jax.random.uniform(pos_rng, (5,), minval=q_min, maxval=q_max)

        # Velocities fixed at zero
        qvel = jax.random.uniform(vel_rng, (5,), minval=-0.0, maxval=0.0)

        # Goal position and orientation fixed at zero
        goal = jax.random.uniform(goal_pos_rng, (2,), minval=-0.0, maxval=0.0)
        mocap_pos = data.mocap_pos.at[0, 0:2].set(goal)
        theta = jax.random.uniform(goal_ori_rng, (), minval=0.0, maxval=0.0)
        mocap_quat = jnp.array([[jnp.cos(theta / 2), 0, 0, jnp.sin(theta / 2)]])

        return data.replace(
            qpos=qpos, qvel=qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat
        )

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe positions relative to the target."""
        pusher_pos = data.qpos[-2:]
        block_pos = data.qpos[0:2]
        block_ori = self.task._get_orientation_err(data)[0:1]
        return jnp.concatenate([pusher_pos, block_pos, block_ori])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 5


class HumanoidEnv(TrainingEnv):
    """Training environment for humanoid (Unitree G1) standup."""

    def __init__(self, episode_length: int) -> None:
        """Set up the walker training environment."""
        super().__init__(task=Humanoid(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, ori_rng = jax.random.split(rng, 4)

        # Random positions and velocities
        qpos = self.task.qstand + 0.1 * jax.random.normal(
            pos_rng, (self.task.model.nq,)
        )
        qvel = 0.1 * jax.random.normal(vel_rng, (self.task.model.nv,))

        # Random base orientation
        u, v, w = jax.random.uniform(ori_rng, (3,))
        quat = jnp.array(
            [
                jnp.sqrt(1 - u) * jnp.sin(2 * jnp.pi * v),
                jnp.sqrt(1 - u) * jnp.cos(2 * jnp.pi * v),
                jnp.sqrt(u) * jnp.sin(2 * jnp.pi * w),
                jnp.sqrt(u) * jnp.cos(2 * jnp.pi * w),
            ]
        )
        qpos = qpos.at[3:7].set(quat)

        return data.replace(qpos=qpos, qvel=qvel)

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the full state, regularized to be agnostic to orientation."""
        height = self.task._get_torso_height(data)[None]
        orientation = self.task._get_torso_orientation(data)  # upright rotation
        return jnp.concatenate([height, orientation, data.qpos[7:], data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 56


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


class CraneEnv(TrainingEnv):
    """Training environment for the luffing crane end-effetor tracking task."""

    def __init__(self, episode_length: int = 100) -> None:
        """Set up the particle training environment."""
        super().__init__(task=Crane(), episode_length=episode_length)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng, target_rng = jax.random.split(rng, 4)

        # Crane state
        # TODO: figure out a more principled way to initialize these.
        # Right now the crane swings wildly at initialization, which prevents
        # gathering a lot of data near the target (which is where the system is
        # mostly at run time).
        q_lim = jnp.array(
            [
                [-1.0, 1.0],  # slew
                [0.0, 1.0],  # luff
                [-1.0, 1.0],  # payload x-pos
                [1.0, 2.2],  # payload y-pos
                [0.3, 1.0],  # payload z-pos
                [1.0, 1.0],  # payload orientation (fixed upright)
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        qpos = self.task.model.qpos0 + jax.random.uniform(
            pos_rng,
            (self.task.model.nq,),
            minval=q_lim[:, 0],
            maxval=q_lim[:, 1],
        )
        qvel = jax.random.uniform(
            vel_rng, (self.task.model.nv,), minval=-0.1, maxval=0.1
        )

        # Target position
        # TODO: figure out a better set of potential target positions
        pos_min = jnp.array([-1.5, 1.0, 0.0])
        pos_max = jnp.array([1.5, 3.0, 1.5])
        target_pos = jax.random.uniform(
            target_rng, (3,), minval=pos_min, maxval=pos_max
        )
        mocap_pos = data.mocap_pos.at[0].set(target_pos)

        # Target orientation - this is unused but must be set so vectorization
        # (which is determined by the size of rng) works properly.
        target_quat = jnp.array([1.0, 0.0, 0.0, 0.0]) + jax.random.uniform(
            target_rng, (4,), minval=-0.0, maxval=0.0
        )
        mocap_quat = data.mocap_quat.at[0].set(target_quat)

        return data.replace(
            qpos=qpos, qvel=qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat
        )

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Observe the full crane state, plus end-effector pos/vel."""
        ee_pos = self.task._get_payload_position(data)
        ee_vel = self.task._get_payload_velocity(data)
        return jnp.concatenate([ee_pos, ee_vel, data.qpos, data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 23
