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
    "Pendulum": [
        "/home/vkurtz/gpc_policies/training_logs/gpc_pendulum/20250426_104155",
        "/home/vkurtz/gpc_policies/training_logs/gpc_pendulum/20250426_104227",
        "/home/vkurtz/gpc_policies/training_logs/gpc_pendulum/20250426_104302",
    ],
    "Cart-Pole": [
        "/home/vkurtz/gpc_policies/training_logs/gpc_cart_pole/20250426_104439",
        "/home/vkurtz/gpc_policies/training_logs/gpc_cart_pole/20250426_104632",
        "/home/vkurtz/gpc_policies/training_logs/gpc_cart_pole/20250426_104858",
    ],
    "Double\nCart-Pole": [
        "/home/vkurtz/gpc_policies/training_logs/gpc_double_cart_pole/20250426_105347",
        "/home/vkurtz/gpc_policies/training_logs/gpc_double_cart_pole/20250426_111134",
        "/home/vkurtz/gpc_policies/training_logs/gpc_double_cart_pole/20250426_112850",
    ],
    "Push-T": [
        "/home/vkurtz/gpc_policies/training_logs/gpc_pusht/20250426_114628",
        "/home/vkurtz/gpc_policies/training_logs/gpc_pusht/20250426_120715",
        "/home/vkurtz/gpc_policies/training_logs/gpc_pusht/20250426_122438",
    ],
    "Walker": [
        "/home/vkurtz/gpc_policies/training_logs/gpc_walker/20250426_124236",
        "/home/vkurtz/gpc_policies/training_logs/gpc_walker/20250426_125828",
        "/home/vkurtz/gpc_policies/training_logs/gpc_walker/20250426_131114",
    ],
    "Crane": [
        "/home/vkurtz/gpc_policies/training_logs/gpc_crane/20250426_132111",
        "/home/vkurtz/gpc_policies/training_logs/gpc_crane/20250426_132949",
        "/home/vkurtz/gpc_policies/training_logs/gpc_crane/20250426_133617",
    ],
    # "Humanoid": "/home/vkurtz/gpc_policies/training_logs/gpc_humanoid/20250117_221301",
}
i = 0
for name, log_files in examples.items():
    # Load data from each of the log files
    for j in range(len(log_files)):
        log_file = log_files[j]
        print(f"Loading {name} data from {log_file}")

        loss = get_data(log_file, "fit/loss")
        frac = get_data(log_file, "sim/policy_best_frac")
        cost = get_data(log_file, "sim/policy_cost")
        iters = np.arange(1, len(loss) + 1)

        # Plot the data
        ax[0, i].plot(iters, cost, linewidth=3)
        ax[1, i].plot(iters, 100 * frac, linewidth=3)
        ax[2, i].plot(iters, loss, linewidth=3)

    # Set formatting and labels
    ax[0, i].set_title(name)
    for j in range(3):
        ax[j, i].set_xticks([iters[0], iters[-1]])
    ax[0, i].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.2f}")
    )
    ax[2, i].set_xlabel("Iteration")
    i += 1

# Set labels for the first column
ax[0, 0].set_ylabel("Cost")
ax[1, 0].set_ylabel("Best %")
ax[2, 0].set_ylabel("Loss")
plt.tight_layout()
plt.show()
