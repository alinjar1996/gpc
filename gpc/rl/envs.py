import jax
from brax.envs.base import PipelineEnv, State

from gpc.envs import TrainingEnv


class BraxEnv(PipelineEnv):
    """Brax wrapper for GPC training environments.

    This will allow us to use standard RL algorithms like PPO to train
    policies with the same dynamics and objective function that we use for GPC.
    """

    def __init__(self, env: TrainingEnv):
        """Create a BRAX environment from a GPC training environment."""
        self.env = env
        self.task = env.task
        self.model = env.task.model

        super().__init__(
            self.model,
            n_frames=self.task.sim_steps_per_control_step,
            backend="mjx",
        )

    def reset(self, rng: jax.Array) -> State:
        """Reset to a fresh initial state."""
        gpc_state = self.env.init_state(rng)
        mjx_data = gpc_state.data

        obs = self.env.get_obs(mjx_data)

        reward = 0.0
        done = 0.0
        metrics = {
            "timestep": gpc_state.t,
            "cost": -reward,
        }
        return State(gpc_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Compute rewards, observations, and the next state."""
        # Advance the system state, resetting the counter (to trigger a terminal
        # cost calculation) at the planning horizon.
        gpc_state = self.env.step(state.pipeline_state, action)
        gpc_state = gpc_state.replace(
            t=gpc_state.t % (self.task.planning_horizon + 1)
        )

        # Reward is given by the terminal cost (often larger) every N steps
        reward = jax.lax.cond(
            gpc_state.t >= self.task.planning_horizon,
            lambda: -self.task.terminal_cost(gpc_state.data),
            lambda: -self.task.dt
            * self.task.running_cost(gpc_state.data, action),
        )

        obs = self.env.get_obs(gpc_state.data)
        state.metrics.update(
            timestep=gpc_state.t,
            cost=-reward,
        )
        return state.replace(
            pipeline_state=gpc_state,
            obs=obs,
            reward=reward,
        )

    @property
    def observation_size(self) -> int:
        """Number of observations."""
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        """Number of actions."""
        return self.task.model.nu
