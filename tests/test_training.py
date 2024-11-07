from pathlib import Path

import jax
import jax.numpy as jnp
from hydrax.tasks.particle import Particle

from gpc.architectures import ActionSequenceMLP
from gpc.dataset import TrainingData
from gpc.training import Policy, train


def test_policy() -> None:
    """Test the policy helper class."""
    rng = jax.random.key(0)
    num_steps = 5
    num_actions = 2
    num_obs = 3

    # Create a toy score MLP
    rng, init_rng = jax.random.split(rng)
    mlp = ActionSequenceMLP([64, 64], num_steps, num_actions)
    params = mlp.init(init_rng, jnp.zeros(num_obs))

    # Create the policy
    u_min = -2 * jnp.ones(num_actions)
    u_max = 0.1 * jnp.ones(num_actions)
    policy = Policy(mlp, params, u_min, u_max)

    # Test running the policy
    y = jnp.ones((num_obs,))
    u = policy.apply(y)
    assert u.shape == (num_steps, num_actions)

    # Save and load the policy
    local_dir = Path("_test_policy")
    local_dir.mkdir(parents=True, exist_ok=True)

    policy.save(local_dir / "policy.pkl")
    del policy

    policy = Policy.load(local_dir / "policy.pkl")

    u2 = policy.apply(y)
    assert jnp.allclose(u2, u)

    # Cleanup
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_train() -> None:
    """Test the training loop with toy data."""
    rng = jax.random.key(0)
    task = Particle()

    # Make some toy data
    rng, obs_rng, act_rng = jax.random.split(rng, 3)
    dataset = TrainingData(
        old_action_sequence=jax.random.uniform(act_rng, (128, 5, 2)),
        new_action_sequence=jnp.zeros((128, 5, 2)),
        observation=jax.random.uniform(obs_rng, (128, 4)),
        state=None,
    )

    # Train the policy
    net = ActionSequenceMLP([64, 64], 5, 2)
    policy = train(dataset, task, net)
    assert isinstance(policy, Policy)

    # Run the policy for a step
    y = jnp.zeros((4,))
    u = policy.apply(y)
    assert u.shape == (5, 2)


if __name__ == "__main__":
    test_policy()
    test_train()
