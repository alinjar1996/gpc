import sys

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling

from gpc.architectures import DenoisingCNN
from gpc.envs import PushTEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = PushTEnv(episode_length=400)
    save_file = "/tmp/pusht_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=128, noise_level=0.4)
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            feature_dims=[32, 32, 32],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=32,
            log_dir="/tmp/gpc_pusht",
            num_iters=20,
            num_envs=128,
            num_epochs=10,
            checkpoint_every=5,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        mj_data = mujoco.MjData(env.task.mj_model)
        mj_data.qpos[:] = [0.1, 0.1, 1.5, 0.0, 0.0]  # set the initial state
        test_interactive(env, policy, mj_data)

    else:
        print(usage)
        sys.exit(1)
