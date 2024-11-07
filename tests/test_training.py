from pathlib import Path

import jax
import jax.numpy as jnp
from hydrax.tasks.particle import Particle

from gpc.architectures import ScoreMLP
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
    mlp = ScoreMLP([64, 64])
    params = mlp.init(
        init_rng, jnp.zeros((num_steps, num_actions)), jnp.zeros((num_obs,))
    )

    # Create the policy
    u_min = -2 * jnp.ones(num_actions)
    u_max = 0.1 * jnp.ones(num_actions)
    policy = Policy(mlp, params, u_min, u_max)

    # Test running the policy
    u_old = jnp.ones((num_steps, num_actions))
    y = jnp.ones((num_obs,))
    u_new = policy.apply(u_old, y)

    # Check that the output is in the correct range
    assert jnp.all(u_new >= u_min)
    assert jnp.all(u_new <= u_max)

    # Save and load the policy
    local_dir = Path("_test_policy")
    local_dir.mkdir(parents=True, exist_ok=True)

    policy.save(local_dir / "policy.pkl")
    del policy

    policy = Policy.load(local_dir / "policy.pkl")

    u_new_2 = policy.apply(u_old, y)
    assert jnp.allclose(u_new, u_new_2)

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
    policy = train(dataset, task)
    assert isinstance(policy, Policy)

    # Run the policy for a step
    u_old = jnp.ones((5, 2))
    y = jnp.zeros((4,))
    u_new = policy.apply(u_old, y)

    assert jnp.linalg.norm(u_new) < jnp.linalg.norm(u_old)


if __name__ == "__main__":
    test_policy()
    test_train()
