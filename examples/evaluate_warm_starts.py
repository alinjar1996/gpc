#!/usr/bin/env python

##
#
# Test performance of trained policies with different warm-starting strategies.
#
##

import pickle
from typing import Tuple

from gpc.envs import (
    CartPoleEnv,
    CraneEnv,
    DoubleCartPoleEnv,
    HumanoidEnv,
    PendulumEnv,
    PushTEnv,
    TrainingEnv,
    WalkerEnv,
)
from gpc.policy import Policy
from gpc.testing import evaluate


def eval_gpc(
    env: TrainingEnv,
    policy_path: str,
    warm_start_level: float,
    use_action_inpainting: bool,
    sim_time: float = 10,
) -> Tuple[float, float]:
    """Evaluate a pretrained policy with the given warm-starting parameters."""
    policy = Policy.load(policy_path)
    num_loops = sim_time // (
        env.task.planning_horizon
        * env.task.sim_steps_per_control_step
        * env.task.model.opt.timestep
    )
    return evaluate(
        env,
        policy,
        num_initial_conditions=100,
        num_loops=int(num_loops),
        warm_start_level=warm_start_level,
        use_action_inpainting=use_action_inpainting,
    )


if __name__ == "__main__":
    # Location of saved policies
    base_dir = "/home/vkurtz/gpc_policies"

    # Define settings for evaluation
    envs = [
        PendulumEnv(200),
        CartPoleEnv(200),
        DoubleCartPoleEnv(400),
        PushTEnv(400),
        WalkerEnv(500),
        CraneEnv(500),
        HumanoidEnv(400),
    ]
    names = [
        "pendulum",
        "cart_pole",
        "double_cart_pole",
        "pusht",
        "walker",
        "crane",
        "humanoid",
    ]
    assert len(envs) == len(names)

    results = {}
    for i in range(len(envs)):
        print(f"Evaluating {names[i]} performance")
        results[names[i]] = {}

        print("==> No warm start")
        mean, std = eval_gpc(
            envs[i],
            f"{base_dir}/{names[i]}_policy.pkl",
            warm_start_level=0.0,
            use_action_inpainting=False,
        )
        results[names[i]]["no_warm_start"] = (mean, std)
        print("")

        print("==> Partial warm start")
        mean, std = eval_gpc(
            envs[i],
            f"{base_dir}/{names[i]}_policy.pkl",
            warm_start_level=0.5,
            use_action_inpainting=False,
        )
        results[names[i]]["partial_warm_start"] = (mean, std)
        print("")

        print("==> Full warm start")
        mean, std = eval_gpc(
            envs[i],
            f"{base_dir}/{names[i]}_policy.pkl",
            warm_start_level=1.0,
            use_action_inpainting=False,
        )
        results[names[i]]["full_warm_start"] = (mean, std)

        print("==> Action inpainting")
        mean, std = eval_gpc(
            envs[i],
            f"{base_dir}/{names[i]}_policy.pkl",
            warm_start_level=0.0,
            use_action_inpainting=True,
        )
        results[names[i]]["action_inpainting"] = (mean, std)

    # Save results
    with open("warm_start_eval_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved results!")
