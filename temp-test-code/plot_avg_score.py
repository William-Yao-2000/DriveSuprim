import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os

# 假设你的字典叫 avg_scores
avg_scores = pkl.load(open('temp-test-code/avg_scores.pkl', 'rb'))

# 定义每张图要画哪些子分数
# 你可以自己改这里，把相关的子分数画在同一张图上
plot_groups = {
    'Collision Related': ['no_at_fault_collisions', 'time_to_collision_within_bound'],
    'Driving Compliance': ['drivable_area_compliance', 'driving_direction_compliance', 'traffic_light_compliance', 'lane_keeping'],
    'Comfort and Progress': ['ego_progress', 'history_comfort', 'pdm_score']
}
# 定义每张图要画哪些子分数
plot_groups = {
    'Collision Related': ['no_at_fault_collisions', 'time_to_collision_within_bound'],
    'Driving Compliance': ['drivable_area_compliance', 'driving_direction_compliance', 'traffic_light_compliance', 'lane_keeping'],
    'Comfort and Progress': ['ego_progress', 'history_comfort', 'pdm_score']
}

# 创建保存目录（如果没有的话）
output_dir = 'temp-test-code'
os.makedirs(output_dir, exist_ok=True)

# 创建一个大的图，大小为 (14, 18)
plt.figure(figsize=(14, 18))

# 绘制每组分数
for idx, (group_name, score_names) in enumerate(plot_groups.items()):
    plt.subplot(3, 1, idx+1)  # 3行1列的子图，idx+1表示子图的位置

    for score_name in score_names:
        if score_name in avg_scores:
            y = avg_scores[score_name]
            x = np.arange(len(y))
            k = 50
            plt.scatter(x[0::k], y[0::k], label=score_name, s=2)

    plt.xlabel('Trajectory Index')
    plt.ylabel('Score')
    plt.title(f'{group_name} Scores Over Trajectories')
    plt.legend()
    plt.grid(True)

# 调整布局以避免重叠
plt.tight_layout()

# 保存图像到本地
output_path = os.path.join(output_dir, 'combined_scores.png')
plt.savefig(output_path)

# 关闭图，释放内存
plt.close()

print(f"所有图已保存到 {output_dir} 文件夹！文件名: combined_scores.png")