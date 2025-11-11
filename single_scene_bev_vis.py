import os
from pathlib import Path
import pickle
from typing import Dict, Any

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.visualization.config import NEW_TAB_10, TRAJECTORY_CONFIG
from navsim.visualization.plots import plot_bev_frame
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def _add_trajectory_to_bev_ax(ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
    """
    Add trajectory poses as lint to plot
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :return: ax with plot
    """
    poses = trajectory.poses[:, :2]
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
    )
    return ax


SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "navtest_vis"

# import pdb; pdb.set_trace()

def changeTraj(traj: Trajectory) -> Trajectory:
    traj.poses = traj.poses[4::5]
    traj.trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    return traj

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

single_stage_config = TRAJECTORY_CONFIG['agent'].copy()
single_stage_config['fill_color'] = "#fb4c14"
single_stage_config['line_color'] = "#fb4c14"
multi_stage_config = TRAJECTORY_CONFIG['agent'].copy()
multi_stage_config['fill_color'] = "#0faffd"
multi_stage_config['line_color'] = "#0faffd"


single_stage_subscores = pickle.load(open(f'{os.getenv("NAVSIM_EXP_ROOT")}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs/hydra_mdp_pp/hydra_img_vit_ssl/epoch=09-step=13300.pkl', 'rb'))
multi_stage_subscores = pickle.load(open(f'{os.getenv("NAVSIM_EXP_ROOT")}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/stage_layers_3-topks_256/epoch=05-step=7980-navtest.pkl', 'rb'))

# token = sys.argv[1]
# assert token is not None, "Token must be provided as a command line argument."
for token in ("28d8f3699547568b", "6363aa6d3d715e03", "a0e9cbedca0a56b7", "9d21f2742b1a5b27", "2f421d857f32510e", "020ba7462c6f52b3"):

    # import pdb; pdb.set_trace()
    print("token:", token)

    scene = scene_loader.get_scene_from_token(token)
    
    frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
    fig, ax = plot_bev_frame(scene, frame_idx)

    single_stage_traj: Trajectory = single_stage_subscores[token]['trajectory']
    multi_stage_traj: Trajectory = multi_stage_subscores[token]['trajectory']

    single_stage_traj = changeTraj(single_stage_traj)
    multi_stage_traj = changeTraj(multi_stage_traj)

    _add_trajectory_to_bev_ax(ax, single_stage_traj, single_stage_config)
    _add_trajectory_to_bev_ax(ax, multi_stage_traj, multi_stage_config)


    file_path = Path(f"/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp_v2/vis/bev/{token}.png")
    os.makedirs(file_path.parent, exist_ok=True)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')