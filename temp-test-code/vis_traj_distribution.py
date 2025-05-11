import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm

# 假设 devkit_root 已经正确设置
devkit_root = os.getenv('NAVSIM_DEVKIT_ROOT')
vocab_size = 8192
traj_path = f"{devkit_root}/traj_final/test_{vocab_size}_kmeans.npy"
counter_path = "temp-test-code/traj_counter_top3_ori.npy"

import pdb; pdb.set_trace()

try:
    vocab = np.load(traj_path)
    counter = np.load(counter_path)
    num_trajectories = vocab.shape[0]

    # 确保 counter 的长度与轨迹数量一致
    if counter.shape[0] != num_trajectories:
        raise ValueError(f"轨迹数量 ({num_trajectories}) 与计数器长度 ({counter.shape[0]}) 不匹配。")

    # 归一化 counter，得到 0 到 1 之间的权重
    norm = Normalize(vmin=counter.min(), vmax=counter.max())
    cmap = cm.get_cmap('coolwarm')  # 使用 'coolwarm' 色图，蓝色到红色

    plt.figure(figsize=(10, 10))
    for i in range(num_trajectories):
        trajectory = vocab[i]
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        color = cmap(norm(counter[i]) ** 0.7)  # 根据权重获取颜色
        plt.plot(x, y, linewidth=0.5, color=color)

    # 添加颜色条
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=plt.gca()
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Visualization of {num_trajectories} Trajectories in XY Plane (Colored by Frequency)")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # 确保 x 和 y 轴的比例一致

    # 保存图形
    os.makedirs("temp-test-code", exist_ok=True)
    plt.savefig("temp-test-code/trajectories_colored.png")

    # 显示图形
    plt.show()

except FileNotFoundError as e:
    print(f"错误：文件未找到，路径为 {e.filename}。请检查路径是否正确。")
except ValueError as e:
    print(f"数值错误：{e}")
except Exception as e:
    print(f"发生了一个错误：{e}")