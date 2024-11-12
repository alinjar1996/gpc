import sys

from hydrax.algs import PredictiveSampling

from gpc.architectures import ActionSequenceMLP
from gpc.envs import WalkerEnv
from gpc.policy import Policy
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
        ctrl = PredictiveSampling(env.task, num_samples=64, noise_level=0.3)
        net = ActionSequenceMLP(
            [128, 128], env.task.planning_horizon, env.task.model.nu
        )
        policy = train(
            env,
            ctrl,
            net,
            log_dir="/tmp/gpc_walker",
            num_policy_samples=64,
            policy_noise_level=0.1,
            num_iters=5,
            num_envs=16,
            num_epochs=50,
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
