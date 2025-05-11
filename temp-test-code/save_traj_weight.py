import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Load the pkl file
seed = 2024
with open(f'/workspace/navtrain-vis-{seed}.pkl', 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()

counter = np.zeros(8192, dtype=np.int32)

for key, value_dict in tqdm(data.items()):
    if 'pdm_score' in value_dict:
        pdm_scores = value_dict['pdm_score']
        if isinstance(pdm_scores, np.ndarray) and pdm_scores.shape == (8192,):
            # Find indices where scores are greater than 0.98
            high_score_indices = np.where(pdm_scores > 0.98)[0]

            if high_score_indices.size > 0:
                # If any scores are greater than 0.98, increment corresponding counters
                counter[high_score_indices] += 1
            else:
                # If no scores exceed 0.98, take top 3 highest scoring indices
                top_3_indices = np.argsort(pdm_scores)[-3:]
                counter[top_3_indices] += 1
        else:
            print(f"Warning: 'pdm_score' in key '{key}' is not a numpy array with shape (8192,). Skipping.")
    else:
        print(f"Warning: Key '{key}' does not contain 'pdm_score'. Skipping.")

import pdb; pdb.set_trace()

filename = f'temp-test-code/traj_counter_top3_rotated-{seed}.npy'
np.save(filename, counter)
print(f"Counter saved to file: {filename}")

pass
