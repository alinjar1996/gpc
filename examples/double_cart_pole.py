import sys

from flax import nnx
from hydrax.algs import PredictiveSampling

from gpc.architectures import DenoisingCNN
from gpc.envs import DoubleCartPoleEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = DoubleCartPoleEnv(episode_length=400)
    save_file = "/tmp/double_cart_pole_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=128, noise_level=0.3)
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            feature_dims=(32, 32, 32),
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=64,
            log_dir="/tmp/gpc_double_cart_pole",
            num_iters=50,
            num_envs=128,
            num_epochs=10,
            exploration_noise_level=0.1,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(
            env, policy, inference_timestep=0.01, warm_start_level=1.0
        )

    else:
        print(usage)
        sys.exit(1)
