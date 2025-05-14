import io
import logging
import os
import pickle
import uuid
from pathlib import Path
import json, copy

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
subscores = pickle.load(open(f'{os.getenv("NAVSIM_EXP_ROOT")}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs/hydra_mdp_pp/hydra_img_vit_ssl/epoch=09-step=13300.pkl', 'rb'))
output_dir = f'{os.getenv("NAVSIM_EXP_ROOT")}/vis/single-stage'
os.makedirs(output_dir, exist_ok=True)

logger = logging.getLogger(__name__)

CONFIG_PATH = "../planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"

norm = plt.Normalize(vmin=0.0, vmax=1.0)
cmap = plt.get_cmap('viridis')


def get_overlay(poses, cam2lidar_rot, cam2lidar_tran, cam_intrin, color=(255, 0, 0, 255)):
    coordinates = np.zeros((3, poses.shape[0]))
    coordinates[0] = poses[:, 0]
    coordinates[1] = poses[:, 1]
    coordinates[2] = 0.0

    lidar2cam_rot = np.linalg.inv(cam2lidar_rot)
    coordinates -= cam2lidar_tran.reshape(-1, 1)
    coordinates = np.dot(lidar2cam_rot, coordinates)
    coordinates = np.dot(cam_intrin, coordinates)
    heights = coordinates[2, :]
    points = view_points(coordinates[:3, :], np.eye(3), normalize=True)
    points[2, :] = heights

    mask = np.ones(points.shape[1], dtype=bool)  # type: ignore
    canvas_size = (1080, 1920)
    mask = np.logical_and(mask, points[0, :] < canvas_size[1] - 1)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[1, :] < canvas_size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 0)

    points = points[:, mask]
    depth = heights[mask]

    points = np.int16(np.round(points[:2, :]))
    depth = np.int16(np.round(depth))
    overlay_img = Image.new("RGBA", (canvas_size[1], canvas_size[0]), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay_img)
    # Populate canvas, use maximum color_value for each bin
    depth_canvas = np.zeros(canvas_size, dtype=np.int16)
    for (col, row), d in zip(points.T, depth):
        depth_canvas[row, col] = d

    depth_canvas = torch.from_numpy(depth_canvas)

    inds = (depth_canvas > 0).nonzero()
    for idx, ind in enumerate(inds):
        y, x = ind
        x, y = x.item(), y.item()
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    return overlay_img


def get_distribution(scores, vocab, gt_traj):
    metrics = ['noc', 'da', 'ttc', 'progress', 'lk', 'total']
    # Define the figure size in inches (540 pixels / 100 dpi = 5.4 inches)
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 10.8))  # 3 plots in a row, 2 rows

    for i, ax in enumerate(axes.flat):
        metric = metrics[i]
        vocab_scores = scores[metric].exp().cpu().numpy()
        # scale imitation scores by 10
        if metric == 'imi':
            vocab_scores *= 10

        line_collection = LineCollection(vocab[..., :2],
                                         colors=[cmap(norm(score)) for score in vocab_scores],
                                         alpha=[1.0 if score > 0.1 else 0.001 for score in vocab_scores])
        ax.set_xlim(-5, 65)
        ax.set_ylim(-25, 25)
        ax.add_collection(line_collection)

        # red line in imi plot is gt traj
        if metric == 'imi':
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], c='r', alpha=1.0)

        ax.set_title(f"Metric {metric}")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]))
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)

    return image


def worker_task(args):
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        import pdb; pdb.set_trace()

    for arg in tqdm(args, desc="Running visualization"):
        token, subscores, vocab = arg['token'], arg['subscores'], arg['vocab']
        scene_loader = arg['scene_loader']
        agent_input = AgentInput.from_scene_dict_list(
            scene_loader.scene_frames_dicts[token],
            scene_loader._original_sensor_path,
            scene_loader._scene_filter.num_history_frames,
            scene_loader._sensor_config
        )
        gt_traj = Scene.from_scene_dict_list(
            scene_loader.scene_frames_dicts[token],
            scene_loader._original_sensor_path,
            scene_loader._scene_filter.num_history_frames,
            10,
            scene_loader._sensor_config
        ).get_future_trajectory(int(4 / 0.5))
        
        # Get rotation angle from aug info and inverse rotate gt trajectory
        rot = 0
        rot_rad = -np.radians(rot)  # Inverse rotation
        cos_rot, sin_rot = np.cos(rot_rad), np.sin(rot_rad)
        rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
        
        # Debug visualization of trajectory rotation
        if arg['cfg'].debug:
            import pdb; pdb.set_trace()
            before_rot = gt_traj.poses[:, :2].copy()
            # gt_traj.poses[:, :2] = np.dot(gt_traj.poses[:, :2], rot_matrix.T)
            after_rot = gt_traj.poses[:, :2]
            
            # plt.figure(figsize=(10, 5))
            # plt.subplot(121)
            # plt.scatter(before_rot[:, 0], before_rot[:, 1], c='b', label='Before rotation')
            # plt.scatter(0, 0, c='k', marker='*', s=200, label='Origin')
            # plt.title(f'Before rotation (rot={rot}°)')
            # plt.axis('equal')
            # plt.grid(True)
            # plt.legend()
            
            # plt.subplot(122) 
            # plt.scatter(after_rot[:, 0], after_rot[:, 1], c='r', label='After rotation')
            # plt.scatter(0, 0, c='k', marker='*', s=200, label='Origin')
            # plt.title(f'After rotation (rot={rot}°)')
            # plt.axis('equal')
            # plt.grid(True)
            # plt.legend()
            
            # plt.savefig(f'{output_dir}/{token}_debug_rotation.png')
            # plt.close()
        else:
            # gt_traj.poses[:, :2] = np.dot(gt_traj.poses[:, :2], rot_matrix.T)
            pass

        subscore = subscores[token]
        for k, v in subscore.items():
            if k != 'trajectory' and k != 'trajectory_pre' and k != 'comfort' and k not in ('lk', 'tl', 'dr') and k != 'filtered_traj':
                subscore[k] = torch.from_numpy(v)

        # inference
        # selected_index = subscore['total'].argmax(-1)

        # model_traj = vocab[selected_index]
        # Get model trajectory and rotate it according to aug rotation
        model_traj = subscore['trajectory'].poses
        model_traj = copy.deepcopy(model_traj)
        rot_rad = np.radians(rot)  # Inverse rotation to match GT trajectory rotation
        cos_rot, sin_rot = np.cos(rot_rad), np.sin(rot_rad)
        rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])
        # if arg['cfg'].debug:
        #     import pdb; pdb.set_trace()
        model_traj[:, :2] = np.dot(model_traj[:, :2], rot_matrix.T)


        # Debug visualization of predicted and GT trajectories
        if arg['cfg'].debug:
            plt.figure(figsize=(5, 5))            
            plt.subplot(111)
            plt.scatter(model_traj[:, 0], model_traj[:, 1], c='b', label='Predicted trajectory')
            plt.scatter(gt_traj.poses[:, 0], gt_traj.poses[:, 1], c='r', label='GT trajectory') 
            plt.scatter(0, 0, c='k', marker='*', s=200, label='Origin')
            plt.title(f'Trajectories after {rot}° rotation')
            plt.axis('equal') 
            plt.grid(True)
            plt.legend()
            
            plt.savefig(f'{output_dir}/{token}_debug_trajectories.png')
            plt.close()

        gt_traj = gt_traj.poses
        file_name = f'{token}'
        save_path = f'{output_dir}/{file_name}.png'
        # if os.path.exists(save_path):
        #     continue

        # inf traj + gt traj
        cam = agent_input.cameras[-1].cam_f0
        img = cam.image
        # Get original transformation matrices
        # orig_cam2lidar_rot = cam.sensor2lidar_rotation 
        # orig_cam2lidar_tran = cam.sensor2lidar_translation
        # orig_cam_intrin = cam.intrinsics

        # Calculate rotation matrix for camera rotation (negative rot since we're calculating camera frame rotation)
        # rot_rad = np.radians(rot)
        # cos_rot = np.cos(rot_rad)
        # sin_rot = np.sin(rot_rad)
        # rot_matrix = np.array([[cos_rot, -sin_rot, 0],
        #                     [sin_rot, cos_rot, 0],
        #                     [0, 0, 1]])

        # # Update cam2lidar rotation by applying additional rotation
        # cam2lidar_rot = orig_cam2lidar_rot @ rot_matrix

        # # Update translation based on rotation
        # rot_matrix_2d = np.array([[cos_rot, -sin_rot],
        #                          [sin_rot, cos_rot]])
        # cam2lidar_tran = orig_cam2lidar_tran.copy()
        # cam2lidar_tran[:2] = rot_matrix_2d @ orig_cam2lidar_tran[:2]

        # # Camera intrinsics matrix remains unchanged
        # cam_intrin = orig_cam_intrin
        # f0 = img
        # l1 = agent_input.cameras[-1].cam_l1.image
        # l0 = agent_input.cameras[-1].cam_l0.image[:, 416:-416]
        # r0 = agent_input.cameras[-1].cam_r0.image[:, 416:-416]
        # r1 = agent_input.cameras[-1].cam_r1.image
        # stitched_image = np.concatenate([l1, l0, f0, r0, r1], axis=1)
        # img_w = l0.shape[1] + f0.shape[1] + r0.shape[1]
        # l1_w = l1.shape[1]
        # r1_w = r1.shape[1]
        # half_view_w = img_w + l1_w // 2 + r1_w // 2
        # whole_w = stitched_image.shape[1]
        # offset_w = int(half_view_w / 180 * rot)
        # stitched_image = stitched_image[:, int(whole_w/2-offset_w-1920/2):int(whole_w/2-offset_w+1920/2)]
        # img = stitched_image
        cam2lidar_rot, cam2lidar_tran, cam_intrin = cam.sensor2lidar_rotation, cam.sensor2lidar_translation, cam.intrinsics

        img = Image.fromarray(img.astype('uint8'), 'RGB').convert('RGBA')

        img = Image.alpha_composite(img, get_overlay(model_traj, cam2lidar_rot, cam2lidar_tran, cam_intrin,
                                                     color=(240, 0, 0, 255)))

        img = Image.alpha_composite(img, get_overlay(gt_traj, cam2lidar_rot, cam2lidar_tran, cam_intrin,
                                                     color=(0, 255, 0, 255)))
        img = img.convert('RGB')

        # # distributions of vocab
        # figs = get_distribution(subscore, vocab, gt_traj)

        # # concat
        # total_width = img.width + figs.width
        # max_height = max(img.height, figs.height)
        # new_image = Image.new('RGB', (total_width, max_height))
        # new_image.paste(img, (0, 0))
        # new_image.paste(figs, (img.width, 0))
        # new_image.save(save_path)

        # concat
        total_width = img.width
        max_height = img.height
        new_image = Image.new('RGB', (total_width, max_height))
        new_image.paste(img, (0, 0))
        # new_image.paste(figs, (img.width, 0))
        new_image.save(save_path)

    return []


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
    # offline_aug_file = cfg.offline_aug_file
    # with open(offline_aug_file, 'r') as f:
    #     aug_info = json.load(f)
    #     aug_info = aug_info['tokens']

    worker = build_worker(cfg)

    data_points = []
    for token in tqdm(scene_loader.tokens):
        data_points.append({
            'cfg': cfg,
            'token': token,
            'scene_loader': scene_loader,
            'vocab': vocab,
            'subscores': subscores,
        })

    worker_map(worker, worker_task, data_points[cfg.start_idx: cfg.end_idx])


if __name__ == "__main__":
    with torch.no_grad():
        main()
