import time
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

from gpc.envs import TrainingEnv
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling


def test_interactive(
    env: TrainingEnv,
    policy: Policy,
    mj_data: mujoco.MjData = None,
    inference_timestep: float = 0.1,
    warm_start_level: float = 1.0,
    use_action_inpainting: bool = False,
) -> None:
    """Test a GPC policy with an interactive simulation.

    Args:
        env: The environment, which defines the system to simulate.
        policy: The GPC policy to test.
        mj_data: The initial state for the simulation.
        inference_timestep: The timestep dt to use for flow matching inference.
        warm_start_level: The warm start level to use for the policy.
        use_action_inpainting: Whether to use action inpainting rather than
        warm-starts.
    """
    rng = jax.random.key(0)
    task = env.task

    # Set up the policy
    policy = policy.replace(dt=inference_timestep)
    policy.model.eval()
    jit_policy = jax.jit(
        partial(policy.apply, warm_start_level=warm_start_level)
    )

    if use_action_inpainting:
        # We'll use action inpainting with exponentially decayed weights,
        # as in https://arxiv.org/pdf/2506.07339.
        start = 1
        end = task.planning_horizon - 2
        weights = jnp.clip(
            (start - 1 - jnp.arange(task.planning_horizon)) / (end - start + 1)
            + 1,
            0,
            1,
        )
        weights *= jnp.expm1(weights) / (jnp.e - 1)

        jit_policy = jax.jit(partial(policy.apply_inpainting, weights=weights))

    # Set up the mujoco simultion
    mj_model = task.mj_model
    if mj_data is None:
        mj_data = mujoco.MjData(mj_model)

    # Initialize the action sequence
    actions = jnp.zeros((task.planning_horizon, task.model.nu))

    # Set up an observation function
    mjx_data = mjx.make_data(task.model)

    @jax.jit
    def get_obs(mjx_data: mjx.Data) -> jax.Array:
        """Get an observation from the mujoco data."""
        mjx_data = mjx.forward(task.model, mjx_data)  # update sites & sensors
        return env.get_obs(mjx_data)

    # Run the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            st = time.time()

            # Get an observation
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
            )
            obs = get_obs(mjx_data)

            # Update the action sequence
            inference_start = time.time()
            rng, policy_rng = jax.random.split(rng)
            actions = jit_policy(prev=actions, y=obs, rng=policy_rng)
            mj_data.ctrl[:] = actions[0]

            inference_time = time.time() - inference_start
            obs_time = inference_start - st
            print(
                f"  Observation time: {obs_time:.5f}s "
                f" Inference time: {inference_time:.5f}s",
                end="\r",
            )

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            elapsed = time.time() - st
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)

    # Save what was last in the print buffer
    print("")


def evaluate(
    env: TrainingEnv,
    policy: Union[Policy, BootstrappedPredictiveSampling],
    num_initial_conditions: int,
    inference_timestep: float = 0.1,
    warm_start_level: float = 1.0,
    use_action_inpainting: bool = False,
    num_loops: int = 1,
    seed: int = 0,
) -> None:
    """Perform a systematic performance evaluation of a GPC policy.

    Runs the policy from a randomized set of initial conditions, and reports
    the total cost of each run.

    Args:
        env: The environment, which defines the system to simulate.
        policy: The GPC policy to test.
        num_initial_conditions: The number of initial conditions to test.
        inference_timestep: The timestep dt to use for flow matching inference.
        warm_start_level: The warm start level to use for the policy.
        use_action_inpainting: Whether to use action inpainting warm-starts.
        num_loops: The number of times to loop through the simulation.
        seed: The random seed to use for the initial conditions.
    """
    rng = jax.random.key(seed)
    task = env.task

    # Set up the policy
    if isinstance(policy, BootstrappedPredictiveSampling):

        def policy_fn(
            data: mjx.Data, action_tape: jax.Array, rng: jax.Array
        ) -> jax.Array:
            """Apply the policy, updating the given action sequence."""
            data = mjx.forward(task.model, data)  # update sites & sensors

            policy_params = policy.init_params()  # valid for PS/MPPI only
            policy_params = policy_params.replace(
                mean=action_tape,
                rng=rng,
            )

            # Do the rollouts, storing the best one in the policy params
            policy_params, _ = policy.optimize(data, policy_params)
            return policy_params.mean

        jit_policy = jax.jit(jax.vmap(policy_fn))

    else:
        policy = policy.replace(dt=inference_timestep)
        policy.model.eval()

        # Weights for action inpainting (if requested)
        start = 1
        end = task.planning_horizon - 2
        weights = jnp.clip(
            (start - 1 - jnp.arange(task.planning_horizon)) / (end - start + 1)
            + 1,
            0,
            1,
        )
        weights *= jnp.expm1(weights) / (jnp.e - 1)

        def policy_fn(
            data: mjx.Data, action_tape: jax.Array, rng: jax.Array
        ) -> jax.Array:
            """Apply the policy, updating the given action sequence."""
            data = mjx.forward(task.model, data)  # update sites & sensors
            obs = env.get_obs(data)
            if use_action_inpainting:
                return policy.apply_inpainting(action_tape, obs, weights, rng)
            return policy.apply(
                action_tape, obs, rng, warm_start_level=warm_start_level
            )

        jit_policy = jax.jit(jax.vmap(policy_fn))

    # Set the initial states
    rng, init_rng = jax.random.split(rng)
    init_rng = jax.random.split(init_rng, num_initial_conditions)
    states = jax.jit(jax.vmap(env.init_state))(init_rng)

    # Set up the simulation step function, x_{t+1} = f(x_t, u_t)
    jit_step = jax.jit(jax.vmap(env.step))

    # Set up the cost functions, l(x_t, u_t)
    jit_running_cost = jax.jit(jax.vmap(env.task.running_cost))
    jit_terminal_cost = jax.jit(jax.vmap(env.task.terminal_cost))

    # Run the simulation
    action_tapes = jnp.zeros(
        (num_initial_conditions, task.planning_horizon, task.model.nu)
    )
    costs = jnp.zeros(num_initial_conditions)
    num_sim_steps = int(task.planning_horizon * task.sim_steps_per_control_step)
    for _ in range(num_loops):
        for _ in range(num_sim_steps):
            print(f"t = {states.data.time[0]:.2f}", end="\r")

            # Get actions from the policy
            action_rng = jax.random.split(rng, num_initial_conditions)
            action_tapes = jit_policy(states.data, action_tapes, action_rng)
            actions = action_tapes[:, 0, :]

            # Evaluate costs at current state
            costs += jit_running_cost(states.data, actions)

            # Advance the state
            states = jit_step(states, actions)

        # Compute the terminal cost
        costs += jit_terminal_cost(states.data)

    # Normalize cost by the number of simulation steps
    costs /= num_sim_steps * num_loops

    # Print performance summary
    final_time = states.data.time[0]
    avg_cost = jnp.mean(costs)
    std_cost = jnp.std(costs)

    print(
        (
            f"Simulated from {num_initial_conditions} initial conditions "
            f"for {final_time:.2f} seconds"
        )
    )
    print(f"Average cost: {avg_cost:.2f} ± {std_cost:.2f}")
    return avg_cost, std_cost
