import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 读取 pkl 文件
with open('/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/dataset/traj_pdm_v2/ori/vocab_score_8192_navtrain/navtrain.pkl', 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()

counter = np.zeros(8192, dtype=np.int32)

for key, value_dict in tqdm(data.items()):
    if 'pdm_score' in value_dict:
        pdm_scores = value_dict['pdm_score']
        if isinstance(pdm_scores, np.ndarray) and pdm_scores.shape == (8192,):
            # 查找所有高于 0.98 的分数的索引
            high_score_indices = np.where(pdm_scores > 0.98)[0]

            if high_score_indices.size > 0:
                # 如果存在高于 0.98 的分数，则将 counter 对应位置加 1
                counter[high_score_indices] += 1
            else:
                # 如果没有高于 0.98 的分数，则取前 3 高的索引
                top_3_indices = np.argsort(pdm_scores)[-3:]
                counter[top_3_indices] += 1
        else:
            print(f"Warning: 'pdm_score' in key '{key}' is not a numpy array with shape (8192,). Skipping.")
    else:
        print(f"Warning: Key '{key}' does not contain 'pdm_score'. Skipping.")

import pdb; pdb.set_trace()

filename = 'temp-test-code/traj_counter_top3_ori.npy'
np.save(filename, counter)
print(f"Counter 已保存到文件: {filename}")

pass