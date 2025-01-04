import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
# )
from flax import nnx
from hydrax.algs import MPPI

from gpc.architectures import DenoisingCNN
from gpc.envs import HumanoidEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = HumanoidEnv(episode_length=200)
    save_file = "/home/vincek/gpc_logs/humanoid_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = MPPI(
            env.task,
            num_samples=128,
            noise_level=1.0,
            num_randomizations=1,
            temperature=0.1,
        )
        net = DenoisingCNN(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=env.task.planning_horizon,
            feature_dims=(64,) * 5,
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=1,
            log_dir="/home/vincek/gpc_logs/humanoid",
            num_iters=100,
            num_envs=512,
            num_epochs=100,
            checkpoint_every=1,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        test_interactive(env, policy, sim_timestep=0.01)

    else:
        print(usage)
        sys.exit(1)
