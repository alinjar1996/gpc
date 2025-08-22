import argparse
import pickle
import time
from pathlib import Path

from brax.training import distribution

from gpc.envs import PendulumEnv
from gpc.rl.envs import BraxEnv
from gpc.rl.ppo import (
    MLP,
    BraxPPONetworksWrapper,
    PolicyWrapper,
    make_policy_function,
    train_ppo,
)
from gpc.testing import test_interactive


def train(path: str) -> None:
    """Train a PPO policy and save it."""
    path = Path(path + "/")
    Path.mkdir(path, exist_ok=True)
    save_path = path / "pendulum_policy.pkl"
    log_path = path / time.strftime("%Y%m%d_%H%M%S")

    # Set up the environment
    episode_length = 200
    env = BraxEnv(PendulumEnv(episode_length))

    # Define value and policy networks
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(16, 2)),
        value_network=MLP(layer_sizes=(16, 1)),
        action_distribution=distribution.NormalTanhDistribution,
    )

    # Train the agent
    train_ppo(
        env=lambda: env,
        network_wrapper=network_wrapper,
        save_path=save_path,
        tensorboard_logdir=log_path,
        num_timesteps=1_000_000,
        num_evals=30,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=1e-3,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        seed=0,
    )


def test(path: str) -> None:
    """Load a trained policy and test it with interactive simulation."""
    env = PendulumEnv(200)

    # Load the trained policy
    save_path = Path(path) / "pendulum_policy.pkl"
    with open(save_path, "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function that looks like GPC (even though it's just a
    # simple feed-forward network)
    policy_fn = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=3,
        action_size=1,
        normalize_observations=True,
        deterministic=True,
    )
    policy = PolicyWrapper(policy_fn)
    test_interactive(env, policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--save_path", type=str, default="/tmp/pendulum_ppo")
    args = parser.parse_args()

    if args.train:
        train(args.save_path)
    elif args.test:
        test(args.save_path)
    else:
        parser.print_help()
