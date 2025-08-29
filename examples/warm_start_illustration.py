##
#
# Illustrate warm-start strategies using a pretrained double cart-pole model.
# Note that the model must be trained first, and saved to
# /tmp/double_cart_pole_policy.pkl.
#
#    python examples/double_cart_pole.py train
#
##

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from gpc.policy import Policy

if __name__ == "__main__":
    rng = jax.random.key(0)

    # Load the policy from a file
    policy = Policy.load("/tmp/double_cart_pole_policy.pkl")
    policy = policy.replace(dt=0.1)
    policy.model.eval()

    # Set an initial observation and action sequence
    obs = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    actions = jnp.zeros((10, 1))

    # Initial policy inference to get the warm-start
    rng, act_rng = jax.random.split(rng)
    warm_start_level = 0.0
    initial_actions = policy.apply(actions, obs, act_rng, warm_start_level)
    initial_actions = policy.apply(
        initial_actions, obs, act_rng, warm_start_level
    )

    # Policy inference without warm-starts
    num_rollouts = 50
    rng, act_rng = jax.random.split(rng)
    act_rng = jax.random.split(act_rng, num_rollouts)
    next_actions_no_ws = jax.vmap(policy.apply, in_axes=(None, None, 0, None))(
        0.0 * initial_actions, obs, act_rng, 0.0
    )

    # Policy inference with warm-starts
    next_actions_full_ws = jax.vmap(
        policy.apply, in_axes=(None, None, 0, None)
    )(initial_actions, obs, act_rng, 0.9)

    # Plot the results
    plt.subplot(2, 1, 1)
    plt.plot(initial_actions, "r-", label="Prev. Actions", linewidth=3)
    for i in range(num_rollouts):
        plt.plot(next_actions_no_ws[i], "k:", alpha=0.5)
    plt.ylim(-1, 1)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(initial_actions, "r-", label="Prev. Actions", linewidth=3)
    for i in range(num_rollouts):
        plt.plot(next_actions_full_ws[i], "k:", alpha=0.5)
    plt.ylim(-1, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()
