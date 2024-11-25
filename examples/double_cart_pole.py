import sys

from flax import nnx
from hydrax.algs import PredictiveSampling

import jax
import jax.numpy as jnp

from gpc.architectures import DenoisingCNN
from gpc.envs import DoubleCartPoleEnv
from gpc.policy import Policy
from gpc.testing import test_interactive
from gpc.training import train

from mujoco import mjx


class GPCSamplingPolicy(PredictiveSampling):
    """Sampling-based controller that takes the best generated control tape."""
    def __init__(self, env, policy, num_samples):
        super().__init__(env.task, num_samples=num_samples, noise_level=0.0)
        self.env = env
        self.policy = policy
        self.policy.model.eval()

    def optimize(self, state, rng):
        # Get random seeds
        rng, sample_rng = jax.random.split(rng)
        sample_rngs = jax.random.split(sample_rng, self.num_samples)

        # Set the initial guess of the action sequence and the observation
        U0 = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        y = self.env.get_obs(state)

        # Generate num_samples action sequences
        Us = jax.vmap(self.policy.apply, in_axes=(None, None, 0))(
            U0, y, sample_rngs)
        
        # Pick the best action sequence based on rollouts
        rng, rollout_rng = jax.random.split(rng)
        rollouts = self.rollout_with_randomizations(state, Us, rollout_rng)
        costs = jnp.sum(rollouts.costs, axis=1)
        best_idx = jnp.argmin(costs)
        U_best = Us[best_idx]
        
        return U_best


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
            num_iters=10,
            num_envs=128,
            num_epochs=10,
            exploration_noise_level=0.0,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif sys.argv[1] == "test":
        # Load the policy from a file and test it interactively
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)

        ctrl = GPCSamplingPolicy(env, policy, num_samples=128)

        rng = jax.random.key(0)
        state = mjx.make_data(env.task.model)
        rng, opt_rng = jax.random.split(rng)
        U = ctrl.optimize(state, opt_rng)



        test_interactive(
          env, ctrl, inference_timestep=0.01, warm_start_level=1.0
        )

    else:
        print(usage)
        sys.exit(1)
