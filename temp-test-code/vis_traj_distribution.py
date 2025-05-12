import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_and_sum_counters(paths):
    total = None
    for path in paths:
        counter = np.load(path)
        total = counter if total is None else total + counter
    return total

def plot_trajectories(ax, vocab, counter_normalized, title, invert_x=True):
    sorted_indices = np.argsort(counter_normalized)
    cmap = cm.get_cmap('coolwarm')
    for i in sorted_indices:
        traj = vocab[i]
        x, y = traj[:, 0], traj[:, 1]
        ax.plot(y, x, linewidth=0.5, color=cmap(counter_normalized[i] ** 2))
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    if invert_x:
        ax.invert_xaxis()

# === Paths ===
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
vocab_size = 8192
traj_path = f"{devkit_root}/traj_final/test_{vocab_size}_kmeans.npy"
vocab = np.load(traj_path)
num_trajectories = vocab.shape[0]

# === Load counters ===
paths_ori = ["temp-test-code/traj_counter_top3_ori.npy"]
paths_rotated = [
    f"temp-test-code/traj_counter_top3_ori.npy",
    f"temp-test-code/traj_counter_top3_rotated-2024.npy",
    # You can uncomment the following if available
    # f"temp-test-code/traj_counter_top3_rotated-2025.npy",
    # f"temp-test-code/traj_counter_top3_rotated-2026.npy"
]

counter_ori = load_and_sum_counters(paths_ori)
counter_rot = load_and_sum_counters(paths_rotated)

# === Sanity check ===
assert counter_ori.shape[0] == num_trajectories
assert counter_rot.shape[0] == num_trajectories

# === Normalize ===
counter_ori = counter_ori / counter_ori.sum()
counter_rot = counter_rot / counter_rot.sum()

norm = Normalize(vmin=0, vmax=1)
counter_ori_norm = counter_ori / counter_ori.max()
counter_rot_norm = counter_rot / counter_rot.max()

# === Plot side-by-side ===
fig, (ax1, ax2, cax) = plt.subplots(
    1, 3,
    figsize=(20, 10),
    gridspec_kw={"width_ratios": [1, 1, 0.05]}
)

plot_trajectories(ax1, vocab, counter_ori_norm, "w.o. augmentation", invert_x=True)
plot_trajectories(ax2, vocab, counter_rot_norm, "w. augmentation", invert_x=True)

# === Shared colorbar ===
cmap = cm.get_cmap('coolwarm')
plt.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax
)

os.makedirs("temp-test-code", exist_ok=True)
plt.savefig("temp-test-code/trajectories_ori_vs_rotated.pdf", bbox_inches='tight')
plt.show()
