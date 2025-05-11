import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Assume devkit_root is properly set
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
vocab_size = 8192
traj_path = f"{devkit_root}/traj_final/test_{vocab_size}_kmeans.npy"

rotated = True
seed = 2024
if rotated:
    counter_path = f"temp-test-code/traj_counter_top3_rotated-{seed}.npy"
else:
    counter_path = "temp-test-code/traj_counter_top3_ori.npy"

try:
    vocab = np.load(traj_path)
    counter = np.load(counter_path)
    num_trajectories = vocab.shape[0]

    if counter.shape[0] != num_trajectories:
        raise ValueError(f"Number of trajectories ({num_trajectories}) does not match counter length ({counter.shape[0]}).")

    # Normalize counter to range 0 to 1
    counter_normalized = (counter - counter.min()) / (counter.max() - counter.min())

    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('coolwarm')

    fig, ax = plt.subplots(figsize=(10, 10))

    # Get the indices of the trajectories sorted by counter values (low to high)
    sorted_indices = np.argsort(counter_normalized)

    # Plot trajectories from low to high counter values
    for i in sorted_indices:
        trajectory = vocab[i]
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        color = cmap(norm(counter_normalized[i]))
        ax.plot(y, x, linewidth=0.5, color=color)  # Swap x and y

    ax.set_xlabel("Y")  # Set X to Y axis
    ax.set_ylabel("X")  # Set Y to X axis
    ax.set_title(f"Visualization of {num_trajectories} Trajectories in XY Plane (Colored by Frequency)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    # Invert y-axis to have it go from high to low
    ax.invert_xaxis()

    # Align colorbar with main plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax
    )

    os.makedirs("temp-test-code", exist_ok=True)
    if rotated:
        plt.savefig("temp-test-code/trajectories_rotated_colored.png", bbox_inches='tight')
    else:
        plt.savefig("temp-test-code/trajectories_colored.png", bbox_inches='tight')
    plt.show()

except FileNotFoundError as e:
    print(f"Error: File not found at {e.filename}. Please check the path.")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
