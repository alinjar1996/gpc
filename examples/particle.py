import sys

from hydrax.algs import PredictiveSampling

from gpc.architectures import ActionSequenceMLP
from gpc.particle import ParticleEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = ParticleEnv(episode_length=100)
    save_file = "/tmp/particle_policy.pkl"

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=8, noise_level=0.1)
        net = ActionSequenceMLP(
            [32, 32], env.task.planning_horizon, env.task.model.nu
        )
        policy = train(env, ctrl, net, num_iters=10, num_envs=128)
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        policy = Policy.load(save_file)
        test_interactive(env, policy)

    else:
        print(usage)
        sys.exit(1)
