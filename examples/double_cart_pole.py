import sys

from hydrax.algs import PredictiveSampling

from gpc.architectures import DenoisingMLP
from gpc.envs import DoubleCartPoleEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = DoubleCartPoleEnv(episode_length=200)
    save_file = "/tmp/double_cart_pole_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=128, noise_level=0.2)
        net = DenoisingMLP([256, 256, 256])
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=16,
            log_dir="/tmp/gpc_double_cart_pole",
            num_iters=3,
            num_envs=128,
            num_epochs=10,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(env, policy, inference_timestep=0.01)

    else:
        print(usage)
        sys.exit(1)
