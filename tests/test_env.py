import jax
import jax.numpy as jnp

from gpc.envs import ParticleEnv


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


if __name__ == "__main__":
    test_particle_env()
