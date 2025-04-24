#!/usr/bin/env python

##
#
# Make nice bar charts of the evaluation results for each example
#
##

import matplotlib.pyplot as plt
import numpy as np

# Set a serif fonts and a larger font size
plt.rcParams.update({"font.size": 16})
plt.rcParams["font.family"] = "serif"

# Labels for each task
labels = [
    "Pendulum",
    "Cart-Pole",
    "Double Cart-Pole",
    "Push-T",
    "Walker",
    "Crane",
    "Humanoid",
]

# Raw data: SPC only
spc_means = [0.92, 2.90, 8.59, 0.69, 1.42, 1.72, 13.43]
spc_stds = [0.43, 0.81, 1.97, 0.56, 1.24, 0.37, 0.48]

# Raw data: GPC only
gpc_means = [0.81, 2.11, 10.50, 0.63, 1.64, 1.63, 16.74]
gpc_stds = [0.34, 0.55, 2.56, 0.68, 1.22, 0.36, 1.16]

# Raw Data: GPC + SPC
both_means = [0.74, 1.96, 7.73, 0.41, 0.72, 1.59, 12.27]
both_stds = [0.29, 0.47, 1.73, 0.25, 0.20, 0.33, 0.41]

# Compute 95% confidence intervals
N = 100  # number of samples

spc_ci = np.array(spc_stds) / np.sqrt(N) * 1.96
gpc_ci = np.array(gpc_stds) / np.sqrt(N) * 1.96
both_ci = np.array(both_stds) / np.sqrt(N) * 1.96

# Make a histogram of the data, with each task as a separate subplot. In each
# subplot, show a bar with whiskers for the confidence interfal for SPC, GPC,
# and Both.
num_tasks = len(labels)
fig, ax = plt.subplots(1, num_tasks, figsize=(20, 5))

ax[0].set_ylabel("Avg. Cost Per Step")
for i in range(num_tasks):
    xlabels = ["SPC", "GPC", "GPC + SPC"]
    means = [spc_means[i], gpc_means[i], both_means[i]]
    CIs = [spc_ci[i], gpc_ci[i], both_ci[i]]
    ax[i].bar(
        xlabels,
        means,
        yerr=CIs,
        capsize=5,
        color=["blue", "green", "red"],
        alpha=0.6,
    )
    ax[i].set_title(labels[i])
    ax[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
