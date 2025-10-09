import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_frame
import sys

SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "navtest_vis"

import pdb; pdb.set_trace()

hydra.initialize(config_path="navsim/planning/script/config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}", # data_path
    openscene_data_root / f"sensor_blobs/{SPLIT}", # original_sensor_path
    scene_filter,
    openscene_data_root / "warmup_two_stage/sensor_blobs", # synthetic_sensor_path
    openscene_data_root / "warmup_two_stage/synthetic_scene_pickles", # synthetic_scenes_path
    sensor_config=SensorConfig.build_all_sensors(),
)

# token = sys.argv[1]
# assert token is not None, "Token must be provided as a command line argument."
for token in ("28d8f3699547568b", "6363aa6d3d715e03", "a0e9cbedca0a56b7", "9d21f2742b1a5b27"):
    scene = scene_loader.get_scene_from_token(token)


    frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
    fig, ax = plot_bev_frame(scene, frame_idx)
    file_path = Path(f"/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp_v2/vis/bev/{token}.png")
    os.makedirs(file_path.parent, exist_ok=True)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')