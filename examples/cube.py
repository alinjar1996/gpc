import sys

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling

from gpc.architectures import DenoisingCNN
from gpc.envs import CubeEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = CubeEnv(episode_length=1_000)
    save_file = "/tmp/cube_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=32, noise_level=0.2)
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            feature_dims=[64, 64, 64],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=32,
            log_dir="/tmp/gpc_cube",
            num_iters=50,
            num_envs=64,
            num_epochs=20,
            checkpoint_every=1,
            num_videos=4,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        mj_data = mujoco.MjData(env.task.mj_model)
        test_interactive(env, policy, mj_data)

    else:
        print(usage)
        sys.exit(1)
