import sys

import mujoco
from flax import nnx
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive as run_sampling

from gpc.architectures import DenoisingCNN
from gpc.envs import WalkerEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = WalkerEnv(episode_length=500)
    save_file = "/tmp/walker_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=16, noise_level=0.3)
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            feature_dims=[64],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            log_dir="/tmp/gpc_walker",
            num_policy_samples=16,
            num_iters=20,
            num_envs=128,
            num_epochs=10,
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

    elif sys.argv[1] == "sample":
        # Use the policy to bootstrap sampling-based MPC
        policy = Policy.load(save_file)
        ctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            warm_start_level=0.5,
            inference_timestep=0.1,
            num_policy_samples=128,
            task=env.task,
            num_samples=0,
            noise_level=0.3,
        )

        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(ctrl, mj_model, mj_data, frequency=50, fixed_camera_id=0)

    else:
        print(usage)
        sys.exit(1)
