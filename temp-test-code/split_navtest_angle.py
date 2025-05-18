import io
import logging
import os
import pickle
import uuid
from pathlib import Path
import json, copy
from collections import defaultdict
from scipy.interpolate import interp1d

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from hydra.utils import instantiate
from matplotlib.collections import LineCollection
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.common.dataclasses import AgentInput, Scene
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.visualization.private import view_points

"""
RUN WITH 
python navtest.py scene_filter=navtest experiment_name=debug split=test worker=ray_distributed_no_torch worker.threads_per_node=16
"""

# your path to these files
vocab = np.load(f'{os.getenv("NAVSIM_DEVKIT_ROOT")}/traj_final/test_8192_kmeans.npy')
subscores = pickle.load(open(f'{os.getenv("NAVSIM_EXP_ROOT")}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/stage_layers_3-topks_256/epoch=05-step=7980.pkl', 'rb'))
output_dir = f'{os.getenv("NAVSIM_EXP_ROOT")}/vis/multi-stage/final'
os.makedirs(output_dir, exist_ok=True)

logger = logging.getLogger(__name__)

CONFIG_PATH = "../navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"

public_dict = {}


def worker_task(args):
    result = []
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        import pdb; pdb.set_trace()

    for arg in tqdm(args, desc="Running visualization"):
        token, subscores, vocab = arg['token'], arg['subscores'], arg['vocab']
        scene_loader = arg['scene_loader']
        gt_traj = Scene.from_scene_dict_list(
            scene_loader.scene_frames_dicts[token],
            scene_loader._original_sensor_path,
            scene_loader._scene_filter.num_history_frames,
            10,
            scene_loader._sensor_config
        ).get_future_trajectory(int(4 / 0.5))

        last_pose = gt_traj.poses[-1]
        x, y = last_pose[0], last_pose[1]
        angle_rad = np.arctan2(y, x)
        angle_deg = np.degrees(angle_rad)
        category = classify_angle(angle_deg)
        result.append({token: category})

    return result



def classify_angle(angle_deg):
    if -15 <= angle_deg <= 15:
        return 1
    elif angle_deg < -15:
        return 0
    else:
        return 2


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    if cfg.debug:
        import pdb; pdb.set_trace()
        os.environ['ROBUST_HYDRA_DEBUG'] = 'true'
    
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=True,
            cam_l0=True,
            cam_l1=True,
            cam_l2=True,
            cam_r0=True,
            cam_r1=True,
            cam_r2=True,
            cam_b0=True,
            lidar_pc=False,
        )
    )


    data_points = []
    for token in tqdm(scene_loader.tokens):
        data_points.append({
            'cfg': cfg,
            'token': token,
            'scene_loader': scene_loader,
            'vocab': vocab,
            'subscores': subscores,
        })

    worker = build_worker(cfg)
    results = worker_map(worker, worker_task, data_points)
    
    import pdb; pdb.set_trace()

    # 合并所有 worker 的返回结果
    public_dict = defaultdict(list)
    for partial_dict in results:
        for k, v in partial_dict.items():
            public_dict[v].append(k)

    # 保存
    output_path = 'temp-test-code/token_angle_categories.json'
    with open(output_path, 'w') as f:
        json.dump(public_dict, f, indent=2)


if __name__ == "__main__":
    with torch.no_grad():
        main()
