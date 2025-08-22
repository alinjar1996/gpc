import argparse
import pickle
import time
from pathlib import Path

from brax.training import distribution

from gpc.envs import HumanoidEnv
from gpc.rl.envs import BraxEnv
from gpc.rl.ppo import (
    MLP,
    BraxPPONetworksWrapper,
    PolicyWrapper,
    make_policy_function,
    train_ppo,
)
from gpc.testing import evaluate, test_interactive


def train(path: str) -> None:
    """Train a PPO policy and save it."""
    path = Path(path + "/")
    Path.mkdir(path, exist_ok=True)
    save_path = path / "humanoid_policy.pkl"
    log_path = path / time.strftime("%Y%m%d_%H%M%S")

    # Set up the environment
    episode_length = 400
    env = BraxEnv(HumanoidEnv(episode_length))

    # Define value and policy networks (match GPC scale for humanoid)
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(128, 128, 2 * env.action_size)),
        value_network=MLP(layer_sizes=(128, 128, 1)),
        action_distribution=distribution.NormalTanhDistribution,
    )

    # Training steps: 64 (samples) x 128 (envs) x 400 (ep len) x 50 (iters)
    num_timesteps = 64 * 128 * 400 * 50  # 81,920,000

    train_ppo(
        env=lambda: env,
        network_wrapper=network_wrapper,
        save_path=save_path,
        tensorboard_logdir=log_path,
        num_timesteps=num_timesteps,
        num_evals=20,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=1e-3,
        entropy_cost=0.0,
        num_envs=128,
        batch_size=128,
        seed=0,
    )


def test(path: str) -> None:
    """Load a trained policy and test it with interactive simulation."""
    env = HumanoidEnv(400)

    # Load the trained policy
    save_path = Path(path) / "humanoid_policy.pkl"
    with open(save_path, "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function that looks like GPC
    policy_fn = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=env.observation_size,
        action_size=env.task.model.nu,
        normalize_observations=True,
        deterministic=True,
    )
    policy = PolicyWrapper(policy_fn)
    test_interactive(env, policy)


def eval(path: str) -> None:
    """Load a trained policy and evaluate it."""
    env = HumanoidEnv(400)

    # Load the trained policy
    save_path = Path(path) / "humanoid_policy.pkl"
    with open(save_path, "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function that looks like GPC
    policy_fn = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=env.observation_size,
        action_size=env.task.model.nu,
        normalize_observations=True,
        deterministic=True,
    )
    policy = PolicyWrapper(policy_fn)

    evaluate(env, policy, num_initial_conditions=100, num_loops=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/vkurtz/gpc_policies/rl_baselines/humanoid_ppo",
    )
    args = parser.parse_args()

    if args.train:
        train(args.save_path)
    elif args.test:
        test(args.save_path)
    elif args.eval:
        eval(args.save_path)
    else:
        parser.print_help()
