import pickle
import torch
import numpy as np
from tqdm import tqdm


# 读取 pkl 文件
with open('/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/dataset/traj_pdm_v2/ori/vocab_score_8192_navtrain/navtrain.pkl', 'rb') as f:
    data = pickle.load(f)

import pdb; pdb.set_trace()

# 用来累加每个 score_name 的所有 tensor
score_sums = {}

for i, (token, scores) in tqdm(enumerate(data.items())):
    sorted_scores = {}
    for score_name, score_array in scores.items():
        # 对每个 numpy array 自身的元素做降序排列
        sorted_array = np.sort(score_array)[::-1]
        sorted_scores[score_name] = sorted_array

    for score_name, sorted_array in sorted_scores.items():
        if score_name not in score_sums:
            score_sums[score_name] = sorted_array.astype(np.float32)
        else:
            score_sums[score_name] += sorted_array.astype(np.float32)
    
    if i == 4000:
        import pdb; pdb.set_trace()
    pass

# 计算每个指标的平均数组
avg_scores = {}
for score_name in score_sums:
    avg_scores[score_name] = score_sums[score_name] / len(data)

# 保存为新的 pkl 文件
with open('avg_scores.pkl', 'wb') as f:
    pickle.dump(avg_scores, f)

print("处理完成，已保存到 avg_scores.pkl")