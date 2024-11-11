import time

import jax
import jax.numpy as jnp
import optax
from hydrax.algs import PredictiveSampling

from gpc.architectures import ActionSequenceMLP
from gpc.augmented import PredictionAugmentedController
from gpc.particle import ParticleEnv
from gpc.training import fit_policy, simulate_episode, train


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


def test_fit() -> None:
    """Test fitting the policy network."""
    rng = jax.random.key(0)

    # Make some fake data
    rng, obs_rng, act_rng = jax.random.split(rng, 3)
    y = jax.random.uniform(obs_rng, (128, 4))
    U = 1.2 + 0.05 * jax.random.uniform(act_rng, (128, 5, 2))

    # Set up the policy network
    net = ActionSequenceMLP([32, 32], 5, 2)
    rng, init_rng = jax.random.split(rng)
    params = net.init(init_rng, jnp.zeros(4))

    # Initialize the optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    batch_size = 64
    num_epochs = 80

    # Fit the policy network
    st = time.time()
    rng, fit_rng = jax.random.split(rng)
    params, opt_state, loss = fit_policy(
        y, U, net, params, optimizer, opt_state, fit_rng, batch_size, num_epochs
    )
    print("Fit time:", time.time() - st)
    assert loss < 0.1

    # Check that we successfully overfit
    U_pred = net.apply(params, y[0])
    assert jnp.allclose(U_pred, U[0], atol=0.5)


def test_train() -> None:
    """Test the training loop."""
    env = ParticleEnv()
    ctrl = PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
    net = ActionSequenceMLP(
        [32, 32], env.task.planning_horizon, env.task.model.nu
    )
    train(env, ctrl, net, num_iters=10, num_envs=128)


if __name__ == "__main__":
    # test_simulate()
    # test_fit()
    test_train()
