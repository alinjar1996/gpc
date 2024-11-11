import jax
import jax.numpy as jnp
from hydrax.algs import PredictiveSampling

from gpc.architectures import ActionSequenceMLP
from gpc.augmented import PredictionAugmentedController
from gpc.particle import ParticleEnv
from gpc.training import simulate_episode, train


def test_simulate() -> None:
    """Test simulating an episode."""
    rng = jax.random.key(0)
    env = ParticleEnv(episode_length=13)
    ctrl = PredictionAugmentedController(
        PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
    )
    net = ActionSequenceMLP(
        [32, 32], env.task.planning_horizon, env.task.model.nu
    )
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, jnp.zeros(env.observation_size))

    rng, episode_rng = jax.random.split(rng)
    y, U, J_best, J_pred = simulate_episode(env, ctrl, net, params, episode_rng)

    assert y.shape == (13, 4)
    assert U.shape == (13, 5, 2)
    assert J_best.shape == (13,)
    assert J_pred.shape == (13,)


def test_train() -> None:
    """Test the training loop."""
    env = ParticleEnv()
    ctrl = PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
    train(env, ctrl)


if __name__ == "__main__":
    test_simulate()
    # test_train()
