import jax

from gpc.envs import ParticleEnv
from gpc.rl import BraxEnv


def test_brax_env() -> None:
    """Smoke test for step and reset."""
    rng = jax.random.key(0)
    brax_env = BraxEnv(ParticleEnv())

    jit_reset = jax.jit(brax_env.reset)
    jit_step = jax.jit(brax_env.step)

    rng, reset_rng = jax.random.split(rng)
    brax_state = jit_reset(rng=reset_rng)

    for _ in range(brax_env.task.planning_horizon):
        rng, act_rng = jax.random.split(rng)
        action = jax.random.uniform(
            act_rng, shape=(2,), minval=-1.0, maxval=1.0
        )
        brax_state = jit_step(brax_state, action)

    # We should get the terminal cost at the last step
    assert brax_state.pipeline_state.t == brax_env.task.planning_horizon
    assert brax_state.reward == -brax_env.task.terminal_cost(
        brax_state.pipeline_state.data
    )

    # Going beyond the planning horizon should reset to t=0 and go back to the
    # running cost
    rng, act_rng = jax.random.split(rng)
    action = jax.random.uniform(act_rng, shape=(2,), minval=-1.0, maxval=1.0)
    brax_state = jit_step(brax_state, action)
    assert brax_state.pipeline_state.t == 0


if __name__ == "__main__":
    test_brax_env()
