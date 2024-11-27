from typing import Tuple

import jax
import jax.numpy as jnp

from gpc.envs import ParticleEnv, SimulatorState


def test_particle_env() -> None:
    """Test the particle environment."""
    rng = jax.random.key(0)
    env = ParticleEnv()

    state = env.init_state(rng)
    assert state.t == 0

    state2 = env._reset_state(state)
    assert jnp.all(state.data.qpos != state2.data.qpos)
    assert jnp.all(state.data.qvel != state2.data.qvel)
    assert jnp.all(state.rng != state2.rng)

    obs = env._get_observation(state)
    assert obs.shape == (4,)

    jit_step = jax.jit(env.step)

    state = jit_step(state, jnp.zeros(2))
    assert state.t == 1

    state = state.replace(t=100)
    assert env.episode_over(state)
    state = jit_step(state, jnp.zeros(2))
    assert state.t == 0


def test_render() -> None:
    """Test rendering the particle environment."""
    rng = jax.random.key(0)
    env = ParticleEnv()

    rng, init_rng = jax.random.split(rng)
    state = env.init_state(init_rng)

    def _step(state: SimulatorState, action: jax.Array) -> Tuple:
        state = env.step(state, action)
        return state, state

    num_steps = 100
    rng, act_rng = jax.random.split(rng)
    actions = jax.random.normal(act_rng, (num_steps, 2))
    _, states = jax.lax.scan(_step, state, actions)

    frames = env.render(states, fps=10)
    assert frames.shape == (10, 3, 240, 320)


if __name__ == "__main__":
    test_particle_env()
    test_render()
