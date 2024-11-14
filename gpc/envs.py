from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.task_base import Task
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.tasks.particle import Particle
from hydrax.tasks.pendulum import Pendulum
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

    def init_state(self, rng: jax.Array) -> SimulatorState:
        """Initialize the simulator state."""
        state = SimulatorState(
            data=mjx.make_data(self.task.model), t=0, rng=rng
        )
        return self._reset_state(state)

    @abstractmethod
    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""

    @abstractmethod
    def get_obs(self, state: mjx.Data) -> jax.Array:
        """Get the observation from the simulator state."""

    @property
    @abstractmethod
    def observation_size(self) -> int:
        """The size of the observation space."""

    def _reset_state(self, state: SimulatorState) -> SimulatorState:
        """Reset the simulator state to start a new episode."""
        rng, reset_rng = jax.random.split(state.rng)
        data = self.reset(state.data, reset_rng)
        return SimulatorState(data=data, t=0, rng=rng)

    def _get_observation(self, state: SimulatorState) -> jax.Array:
        """Get the observation from the simulator state."""
        return self.get_obs(state.data)

    def episode_over(self, state: SimulatorState) -> bool:
        """Check if the episode is over.

        Override this method if the episode should terminate early.
        """
        return state.t >= self.episode_length

    def step(self, state: SimulatorState, action: jax.Array) -> SimulatorState:
        """Take a simulation step.

        Args:
            state: The simulator state.
            action: The action to take.

        Returns:
            The new simulator state and the new time step.
        """
        return jax.lax.cond(
            self.episode_over(state),
            lambda _: self._reset_state(state),
            lambda _: state.replace(
                data=mjx.step(self.task.model, state.data.replace(ctrl=action)),
                t=state.t + 1,
            ),
            operand=None,
        )


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
        super().__init__(task=Pendulum(), episode_length=episode_length)
        # TODO: set planning horizon in task constructor
        self.task.planning_horizon = 10

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        rng, pos_rng, vel_rng = jax.random.split(rng, 3)
        qpos = jax.random.uniform(pos_rng, (1,), minval=-0.1, maxval=0.1)
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
        qvel = jax.random.uniform(vel_rng, (3,), minval=-2.0, maxval=2.0)
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
        return jnp.concatenate([jnp.delete(data.qpos, 1), data.qvel])

    @property
    def observation_size(self) -> int:
        """The size of the observation space."""
        return 17
