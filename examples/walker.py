import sys

from hydrax.algs import PredictiveSampling

from gpc.architectures import DenoisingMLP
from gpc.envs import WalkerEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

if __name__ == "__main__":
    usage = f"Usage: python {sys.argv[0]} [train|test]"

    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    env = WalkerEnv(episode_length=200)
    save_file = "/tmp/walker_policy.pkl"

    # DEBUG
    import time

    import jax
    import mujoco
    import mujoco.viewer
    from mujoco import mjx

    data = mjx.make_data(env.task.model)

    mj_model = env.task.mj_model
    mj_data = mujoco.MjData(mj_model)

    rng = jax.random.key(0)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            rng, reset_rng = jax.random.split(rng)
            data = env.reset(data, reset_rng)
            print("reset!")

            mj_data.qpos = data.qpos
            mj_data.qvel = data.qvel
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            time.sleep(1.0)

    print(data.qpos)
    breakpoint()

    if sys.argv[1] == "train":
        # Train the policy and save it to a file
        ctrl = PredictiveSampling(env.task, num_samples=16, noise_level=0.3)
        net = DenoisingMLP([128, 128])
        policy = train(
            env,
            ctrl,
            net,
            log_dir="/tmp/gpc_walker",
            num_policy_samples=16,
            num_iters=10,
            num_envs=128,
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
