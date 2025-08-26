#!/usr/bin/env python

##
#
# Make bar charts showing performance, as recorded by
# `examples/evaluate_performance.py`
#
##

import pickle

import matplotlib.pyplot as plt
import numpy as np

# Set some plotting parameters
plt.rcParams.update({"font.size": 16})
plt.rcParams["font.family"] = "serif"

# Load saved results
with open("evaluation_results.pkl", "rb") as f:
    results = pickle.load(f)

# Customize the titles
titles = {
    "pendulum": "Pendulum",
    "cart_pole": "Cart-Pole",
    "double_cart_pole": "Double Cart-Pole",
    "walker": "Walker",
    "pusht": "Push-T",
    "crane": "Crane",
    "humanoid": "Humanoid",
}

# Set up the figure
num_tasks = len(results.keys())
fig, ax = plt.subplots(1, num_tasks, figsize=(20, 5))

ax[0].set_ylabel("Cost Per Step")
for i in range(num_tasks):
    task = list(results.keys())[i]
    methods = list(results[task].keys())
    means = [results[task][m][0] for m in methods]
    stds = [results[task][m][1] for m in methods]

    x = np.arange(len(methods))
    ax[i].bar(x, means, yerr=stds, capsize=7, color=["C0", "C1", "C2", "C3"])
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(methods)
    ax[i].set_title(titles[task])
    ax[i].set_ylim(bottom=0)
    ax[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
