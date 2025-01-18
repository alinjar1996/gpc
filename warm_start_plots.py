#!/usr/bin/env python3

# Make plots of double pendulum swingup with and without warm starts
import matplotlib.pyplot as plt
import numpy as np

# Larger font size
plt.rcParams.update({"font.size": 20})

# Use serif font
plt.rcParams.update({"font.family": "serif"})

# Load the data
num_steps = 600
tip_heights_0 = np.load("tip_heights_0.npy")[:num_steps]
times_0 = np.load("times_0.npy")[:num_steps]
controls_0 = np.load("controls_0.npy")[:num_steps]

tip_heights_1 = np.load("tip_heights_1.npy")[:num_steps]
times_1 = np.load("times_1.npy")[:num_steps]
controls_1 = np.load("controls_1.npy")[:num_steps]

fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

ax[0, 0].plot(times_0, controls_0, "k-", linewidth=3)
ax[0, 0].set_ylabel("Control")

ax[1, 0].plot(times_0, tip_heights_0, "k-", linewidth=3)
ax[1, 0].set_ylabel("Tip Height")
ax[1, 0].set_xlabel("Time (s)")

ax[0, 1].plot(times_1, controls_1, "k-", linewidth=3)
ax[1, 1].plot(times_1, tip_heights_1, "k-", linewidth=3)
ax[1, 1].set_xlabel("Time (s)")

ax[0, 0].set_title("No Warm Start")
ax[0, 1].set_title("With Warm Start")

plt.tight_layout()
plt.show()
