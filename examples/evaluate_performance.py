#!/usr/bin/env python

##
#
# Test performance of trained SPC, GPC, GPC+, and PPO policies, recording
# the results for later plotting.
#
##

import pickle
from typing import Tuple

from gpc.envs import PendulumEnv, TrainingEnv
from gpc.policy import Policy
from gpc.rl.ppo import PolicyWrapper, make_policy_function
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.testing import evaluate


def eval_ppo(
    env: TrainingEnv, policy_path: str, sim_time: float = 10
) -> Tuple[float, float]:
    """Evaluate a PPO policy."""
    with open(policy_path, "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function that looks like GPC
    policy_fn = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=env.observation_size,
        action_size=env.task.model.nu,
        normalize_observations=True,
        deterministic=True,
    )
    policy = PolicyWrapper(policy_fn)

    # Evaluate the policy
    num_loops = sim_time // (
        env.task.planning_horizon
        * env.task.sim_steps_per_control_step
        * env.task.model.opt.timestep
    )
    return evaluate(
        env, policy, num_initial_conditions=100, num_loops=num_loops
    )


def eval_gpc(
    env: TrainingEnv, policy_path: str, sim_time: float = 10
) -> Tuple[float, float]:
    """Evaluate a GPC policy."""
    policy = Policy.load(policy_path)
    num_loops = sim_time // (
        env.task.planning_horizon
        * env.task.sim_steps_per_control_step
        * env.task.model.opt.timestep
    )
    return evaluate(
        env, policy, num_initial_conditions=100, num_loops=int(num_loops)
    )


def eval_gpc_plus(
    env: TrainingEnv,
    policy_path: str,
    sim_time: float = 10,
    noise_level: float = 0.1,
) -> Tuple[float, float]:
    """Evalutate a GPC policy used to bootstrap SPC."""
    policy = Policy.load(policy_path)
    ctrl = BootstrappedPredictiveSampling(
        policy,
        env.get_obs,
        num_policy_samples=64,
        task=env.task,
        num_samples=64,
        noise_level=noise_level,
    )
    num_loops = sim_time // (
        env.task.planning_horizon
        * env.task.sim_steps_per_control_step
        * env.task.model.opt.timestep
    )
    return evaluate(
        env, ctrl, num_initial_conditions=100, num_loops=int(num_loops)
    )


def eval_spc(
    env: TrainingEnv,
    policy_path: str,
    sim_time: float = 10,
    noise_level: float = 0.1,
) -> Tuple[float, float]:
    """Evalutate a SPC (predictive sampling, policy is unused)."""
    policy = Policy.load(policy_path)
    ctrl = BootstrappedPredictiveSampling(
        policy,
        env.get_obs,
        num_policy_samples=0,
        task=env.task,
        num_samples=128,
        noise_level=noise_level,
    )
    num_loops = sim_time // (
        env.task.planning_horizon
        * env.task.sim_steps_per_control_step
        * env.task.model.opt.timestep
    )
    return evaluate(
        env, ctrl, num_initial_conditions=100, num_loops=int(num_loops)
    )


if __name__ == "__main__":
    env = PendulumEnv(200)

    base_dir = "/home/vkurtz/gpc_policies"

    mean, std = eval_ppo(
        env, f"{base_dir}/rl_baselines/pendulum_ppo/pendulum_policy.pkl"
    )
    mean, std = eval_gpc(env, f"{base_dir}/pendulum_policy.pkl")
    mean, std = eval_gpc_plus(env, f"{base_dir}/pendulum_policy.pkl")
    mean, std = eval_spc(env, f"{base_dir}/pendulum_policy.pkl")
    print(mean, std)
