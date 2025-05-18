import os
import numpy as np
import matplotlib.pyplot as plt

# Load trajectory data
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
vocab_size = 8192
traj_path = f"{devkit_root}/traj_final/test_{vocab_size}_kmeans.npy"
vocab = np.load(traj_path)

# Generate random colors for each trajectory
num_trajectories = vocab.shape[0]

# Plot trajectories
fig, ax = plt.subplots(figsize=(10, 10))

for i in range(num_trajectories):
    traj = vocab[i]
    x = traj[:, 0]
    y = traj[:, 1]
    ax.plot(x, y, linewidth=1.0)

# Remove axis and grid
ax.axis('off')  # Hides all axes and ticks

# Save and show
os.makedirs("temp-test-code", exist_ok=True)
plt.savefig("temp-test-code/all_trajectories_no_axis.png", bbox_inches='tight', pad_inches=0, dpi=400)
plt.show()
