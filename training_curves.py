#!/usr/bin/env python

##
# Plot training curves
##

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea


def get_data(log_file, name):
    """Get the values of a particular scalar from a TensorBoard log file"""
    ea_ = ea.EventAccumulator(log_file)
    ea_.Reload()
    data = ea_.Scalars(name)
    return np.array([x.value for x in data])


# Set up the plot
plt.rcParams.update({"font.size": 18})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(3, 7, figsize=(20, 8))

# Choose which logs to use
examples = {
    "Pendulum": "/tmp/gpc_pendulum/20250119_163602",
    "Cart-Pole": "/tmp/gpc_cart_pole/20250119_162857",
    "Double Cart-Pole": "/tmp/gpc_double_cart_pole/20250118_100315",
    "Push-T": "/tmp/gpc_pusht/20250117_100554",
    "Walker": "/tmp/gpc_walker/20250117_102951",
    "Crane": "/tmp/gpc_crane/20250117_145939",
    "Humanoid": "/tmp/gpc_humanoid/20250117_221301",
}

i = 0
for name, log_file in examples.items():
    print(f"Loading {name} data from {log_file}")
    loss = get_data(log_file, "fit/loss")
    frac = get_data(log_file, "sim/policy_best_frac")
    cost = get_data(log_file, "sim/policy_cost")
    iters = np.arange(1, len(loss) + 1)

    ax[0, i].set_title(name)
    ax[0, i].plot(iters, loss, "k-", linewidth=3)

    ax[1, i].plot(iters, 100 * frac, "k-", linewidth=3)
    ax[1, i].set_ylim([0, 100])

    ax[2, i].plot(iters, cost, "k-", linewidth=3)

    for j in range(3):
        ax[j, i].set_xticks([iters[0], iters[-1]])

    ax[0, i].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.2f}")
    )
    ax[2, i].set_xlabel("Iteration")

    i += 1

ax[0, 0].set_ylabel("Loss")
ax[1, 0].set_ylabel("Best %")
ax[2, 0].set_ylabel("Cost")

plt.tight_layout()
plt.show()


# # Load the data
# loss = get_data(log_file, "fit/loss")
# plt.plot(loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.show()
