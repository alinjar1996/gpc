import sys

from flax import nnx
from hydrax.algs import PredictiveSampling

from gpc.architectures import DenoisingCNN
from gpc.envs import CartPoleEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = CartPoleEnv(episode_length=200)
    save_file = "/tmp/cart_pole_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=32, noise_level=0.1)
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            feature_dims=[32, 32],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=16,
            log_dir="/tmp/gpc_cart_pole",
            num_iters=10,
            num_envs=128,
            num_epochs=100,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(env, policy)

    else:
        print(usage)
        sys.exit(1)
