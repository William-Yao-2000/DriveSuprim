import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm

# Assume devkit_root is properly set
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
vocab_size = 8192
traj_path = f"{devkit_root}/traj_final/test_{vocab_size}_kmeans.npy"
counter_path = "temp-test-code/traj_counter_top3_ori.npy"

import pdb; pdb.set_trace()

try:
    vocab = np.load(traj_path)
    counter = np.load(counter_path)
    num_trajectories = vocab.shape[0]

    # Ensure the length of the counter matches the number of trajectories
    if counter.shape[0] != num_trajectories:
        raise ValueError(f"Number of trajectories ({num_trajectories}) does not match counter length ({counter.shape[0]}).")

    # Normalize the counter to get weights between 0 and 1
    norm = Normalize(vmin=counter.min(), vmax=counter.max())
    cmap = cm.get_cmap('coolwarm')  # Use 'coolwarm' colormap (blue to red)

    plt.figure(figsize=(10, 10))
    for i in range(num_trajectories):
        trajectory = vocab[i]
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        color = cmap(norm(counter[i]) ** 0.7)  # Get color based on weight
        plt.plot(x, y, linewidth=0.5, color=color)

    # Add color bar
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=plt.gca()
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Visualization of {num_trajectories} Trajectories in XY Plane (Colored by Frequency)")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio for x and y axes

    # Save the plot
    os.makedirs("temp-test-code", exist_ok=True)
    plt.savefig("temp-test-code/trajectories_colored.png")

    # Show the plot
    plt.show()

except FileNotFoundError as e:
    print(f"Error: File not found at {e.filename}. Please check the path.")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
