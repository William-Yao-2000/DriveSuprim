import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from math import atan2, degrees

# Assume devkit_root is properly set
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
vocab_size = 8192
traj_path = f"{devkit_root}/traj_final/test_{vocab_size}_kmeans.npy"
counter_path = "temp-test-code/traj_counter_top3_ori.npy"

# Function to load and sum counters for data augmentation cases
def load_and_sum_counters(paths):
    total = None
    for path in paths:
        counter = np.load(path)
        total = counter if total is None else total + counter
    return total

# Load the data
vocab = np.load(traj_path)
counter = np.load(counter_path)

# Load the data augmentation counter data
paths_augmented = [
    f"temp-test-code/traj_counter_top3_ori.npy",
    f"temp-test-code/traj_counter_top3_rotated-2024.npy",
    # You can uncomment the following if available
]
counter_aug = load_and_sum_counters(paths_augmented)

# Ensure the counter matches the number of trajectories
num_trajectories = vocab.shape[0]
if counter.shape[0] != num_trajectories:
    raise ValueError(f"Number of trajectories ({num_trajectories}) does not match counter length ({counter.shape[0]}).")

angle_bound = 60

# Calculate the angles and distribute them into bins of 6° intervals
angle_bins = np.arange(-angle_bound, angle_bound+1, 6)  # Bins from -90° to 90°, with a 6° interval
angle_hist = np.zeros(len(angle_bins)-1)
angle_hist_aug = np.zeros(len(angle_bins)-1)  # For augmented counter
num_traj_in_bins = np.zeros(len(angle_bins)-1)

# Process original counter
for i in range(num_trajectories):
    trajectory = vocab[i]
    x_end = trajectory[-1, 0]  # Last x-coordinate of the trajectory
    y_end = trajectory[-1, 1]  # Last y-coordinate of the trajectory
    
    # Calculate the angle of the trajectory's last point with respect to the origin
    angle = degrees(atan2(y_end, x_end))  # Convert from radians to degrees
    
    # Make sure the angle is between -180 and 180 degrees
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    
    # Only consider angles between -90° and 90°
    if -angle_bound <= angle <= angle_bound:
        # Find the corresponding bin for this angle
        bin_index = np.digitize(angle, angle_bins) - 1
        
        # Add the counter value to the corresponding bin
        angle_hist[bin_index] += counter[i]
        num_traj_in_bins[bin_index] += 1

# Process augmented counter
for i in range(num_trajectories):
    trajectory = vocab[i]
    x_end = trajectory[-1, 0]  # Last x-coordinate of the trajectory
    y_end = trajectory[-1, 1]  # Last y-coordinate of the trajectory
    
    # Calculate the angle of the trajectory's last point with respect to the origin
    angle = degrees(atan2(y_end, x_end))  # Convert from radians to degrees
    
    # Make sure the angle is between -180 and 180 degrees
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    
    # Only consider angles between -90° and 90°
    if -angle_bound <= angle <= angle_bound:
        # Find the corresponding bin for this angle
        bin_index = np.digitize(angle, angle_bins) - 1
        
        # Add the augmented counter value to the corresponding bin
        angle_hist_aug[bin_index] += counter_aug[i]
        num_traj_in_bins[bin_index] += 1

# Normalize the histograms
for j in range(len(angle_hist)):
    angle_hist[j] = angle_hist[j] / num_traj_in_bins[j]

for j in range(len(angle_hist_aug)):
    angle_hist_aug[j] = angle_hist_aug[j] / num_traj_in_bins[j]

# Normalize both histograms by the sum of all values
angle_hist_sum = angle_hist.sum()
angle_hist_aug_sum = angle_hist_aug.sum()

for j in range(len(angle_hist)):
    angle_hist[j] = angle_hist[j] / angle_hist_sum

for j in range(len(angle_hist_aug)):
    angle_hist_aug[j] = angle_hist_aug[j] / angle_hist_aug_sum

# Plot the histograms
plt.figure(figsize=(12, 7))

# Improved color scheme with better contrast and vibrancy
bar_width = 6
plt.bar(angle_bins[:-1], angle_hist, width=bar_width, align='edge', color='royalblue',
        edgecolor='lightgray', label='Original', alpha=0.8, linewidth=0.8)
plt.bar(angle_bins[:-1], angle_hist_aug, width=bar_width, align='edge', color='orange',
        edgecolor='lightgray', label='Augmented', alpha=0.36, linewidth=0.8)

# Labels and title with improved font sizes and styles
plt.xlabel("Angle (degrees)", fontsize=14, fontweight='normal', fontname='DejaVu Sans')
plt.ylabel("Relative Frequency", fontsize=14, fontweight='normal', fontname='DejaVu Sans')
plt.title("Trajectory Angle Distribution (Original v.s. Augmented)", fontsize=16, fontweight='bold', fontname='DejaVu Sans')

# Adjust tick marks for better readability
plt.xticks(np.arange(-angle_bound, angle_bound+1, 15), fontsize=12, fontname='DejaVu Sans')
plt.yticks(fontsize=12, fontname='DejaVu Sans')

# Add a legend and adjust its position (inside the chart if possible)
plt.legend(fontsize=12, loc='upper left', frameon=False, prop={'family': 'DejaVu Sans'})

# Remove grid lines for a cleaner look
plt.grid(axis='y', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()

# Save the plot as a vector graphic
os.makedirs("temp-test-code", exist_ok=True)
plt.savefig("temp-test-code/angle_distribution_comparison_improved.pdf", bbox_inches='tight')