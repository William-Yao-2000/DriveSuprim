import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os


multi_stage_info = pkl.load(open("/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp_v2/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/stage_layers_3-topks_256/epoch=05-step=7980-navtest_vis_1.pkl", 'rb'))
single_stage_info = pkl.load(open("/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp_v2/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs/hydra_mdp_pp/hydra_img_vit_ssl/epoch=09-step=13300.pkl", 'rb'))
gt_info = pkl.load(open("/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/dataset/traj_pdm_v2/ori/vocab_score_8192_navtest_vis_1/navtest_vis_1.pkl", 'rb'))

# scene_token = '9d21f2742b1a5b27'
# scene_token = '2f421d857f32510e'
# scene_token = 'a0e9cbedca0a56b7'
# scene_token = '28d8f3699547568b'
scene_token = '020ba7462c6f52b3'

multi_stage_info = multi_stage_info[scene_token]
single_stage_info = single_stage_info[scene_token]
gt_info = gt_info[scene_token]

filtered_index = multi_stage_info['filtered_index']

gt_scores = gt_info['pdm_score'][filtered_index]


# import pdb; pdb.set_trace()


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 防止溢出
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

result = single_stage_info
single_stage_scores = (
    0.02 * np.log(softmax(result['imi'])) +
    0.1 * np.log(sigmoid(result['traffic_light_compliance'])) +
    0.5 * np.log(sigmoid(result['no_at_fault_collisions'])) +
    0.5 * np.log(sigmoid(result['drivable_area_compliance'])) +
    0.3 * np.log(sigmoid(result['driving_direction_compliance'])) +
    6.0 * np.log(
        5.0 * sigmoid(result['time_to_collision_within_bound']) +
        5.0 * sigmoid(result['ego_progress']) +
        2.0 * sigmoid(result['lane_keeping'])
    )
)

single_stage_scores = single_stage_scores[filtered_index]

# import pdb; pdb.set_trace()


# 如果是 torch tensor，转 numpy
gt_scores = gt_scores.numpy() if hasattr(gt_scores, "numpy") else gt_scores
single_stage_scores = single_stage_scores.numpy() if hasattr(single_stage_scores, "numpy") else single_stage_scores

# 降序排序 gt_scores，并同步排序 single_stage_scores
sorted_indices = np.argsort(-gt_scores)
gt_scores = gt_scores[sorted_indices][:16]
single_stage_scores = single_stage_scores[sorted_indices][:16]

x = np.arange(len(gt_scores))

# 归一化 Single_stage_scores 到 [0.5, 1.5]
ms_min = single_stage_scores.min() - 0.3
ms_max = single_stage_scores.max() + 0.1
single_stage_norm = 0.5 + (single_stage_scores - ms_min) / (ms_max - ms_min + 1e-8)

# 创建图
fig, ax1 = plt.subplots(figsize=(4, 3))

# 左轴：GT Score（原始）
ax1.set_xlabel('Index')
ax1.set_ylabel('GT Score', color='green')
ax1.scatter(x, gt_scores, color='green', label='GT Score', s=20, marker='x')
ax1.tick_params(axis='y', labelcolor='green')
ax1.set_ylim(gt_scores.min()-0.3, gt_scores.max()+0.05)

# 右轴：Single-stage Score（归一化后画图，但显示原始值）
ax2 = ax1.twinx()
ax2.set_ylabel('Single-stage Score', color='red')
ax2.scatter(x, single_stage_norm, color='red', label='Multi-stage Score', s=20, marker='x')
ax2.tick_params(axis='y', labelcolor='red')

# 设置右轴 tick 为 0.5 到 1.5 中的等距值，对应原始值
yticks = np.linspace(0, 1.5, 5)
ytick_labels = [f"{ms_min + y * (ms_max - ms_min):.2f}" for y in yticks]
ax2.set_yticks(yticks)
ax2.set_yticklabels(ytick_labels)

# 图标题和保存
plt.title('GT vs Single-Stage Scores (GT: Raw, Single-Stage: Normalized)')
fig.tight_layout()

save_path = f'temp-test-code/{scene_token}_single_cmp_with_gt.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.close()

# import pdb; pdb.set_trace()

