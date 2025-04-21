import argparse

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingMLP
from gpc.envs import CartPoleEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import evaluate, test_interactive
from gpc.training import train

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Balance an inverted pendulum on a cart"
    )
    subparsers = parser.add_subparsers(
        dest="task", help="What to do (choose one)"
    )
    subparsers.add_parser("train", help="Train (and save) a generative policy")
    subparsers.add_parser("test", help="Test a generative policy")
    subparsers.add_parser("eval", help="Evalute a generative policy")
    subparsers.add_parser(
        "sample", help="Bootstrap sampling-based MPC with a generative policy"
    )
    args = parser.parse_args()

    # Set up the environment and save file
    env = CartPoleEnv(episode_length=200)
    save_file = "/tmp/cart_pole_policy.pkl"

    if args.task == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
        net = DenoisingMLP(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            hidden_layers=[64, 64],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=2,
            log_dir="/tmp/gpc_cart_pole",
            num_iters=10,
            num_envs=128,
            num_epochs=100,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif args.task == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(env, policy)

    elif args.task == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            observation_fn=env.get_obs,
            num_policy_samples=2,
            task=env.task,
            num_samples=1,
            noise_level=0.1,
        )
        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=50)

    elif args.task == "eval":
        # Load the policy from a file and evaluate it
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            num_policy_samples=0,
            task=env.task,
            num_samples=128,
            noise_level=0.1,
        )
        # evaluate(env, policy, num_initial_conditions=100, num_loops=10)
        evaluate(env, ctrl, num_initial_conditions=100, num_loops=10)

    else:
        parser.print_help()
