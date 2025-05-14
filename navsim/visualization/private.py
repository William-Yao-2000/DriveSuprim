from tqdm import tqdm
import traceback
import pickle
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import numpy as np
from navsim.common.dataclasses import SensorConfig
from pathlib import Path
from typing import Dict
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import numpy.typing as npt
import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory, SceneFilter
from navsim.common.dataloader import SceneLoader


logger = logging.getLogger(__name__)

CONFIG_PATH = "../planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle_ddp"


def view_points(
    points: npt.NDArray[np.float64], view: npt.NDArray[np.float64], normalize: bool
) -> npt.NDArray[np.float64]:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

input = 'vov+davit+moe-submission'
output = 'vis_private_davit+vov+moe'

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_filter = instantiate(cfg.scene_filter)
    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs_path,
        sensor_config=SensorConfig.build_all_sensors()
    )
    trajs = pickle.load(open(f'/mnt/c/Users/Administrator/Downloads/submissions/{input}/submission.pkl','rb'))['predictions']

    with open('/mnt/g/navsim_challenge_scripts/competition_in_public_set.txt', 'r') as f:
        public_tokens = f.readlines()
    # print(len(public_tokens), public_tokens[0])
    # print(len(set(input_loader.tokens)&set(public_tokens)))
    # private_tokens = list(set(input_loader.tokens) - set(public_tokens))
    # print(len(private_tokens))
    for token in tqdm(input_loader.tokens, desc="Running evaluation"):
        agent_input = \
            input_loader.get_agent_input_from_token(token)
        
        # todo visualize traj
        curr_traj = trajs[token].poses
        cam = agent_input.cameras[-1].cam_f0
        img, cam2lidar_rot, cam2lidar_tran, cam_intrin = cam.image, cam.sensor2lidar_rotation, cam.sensor2lidar_translation, cam.intrinsics
        coordinates = np.zeros((3, 40))
        coordinates[0] = curr_traj[:, 0]
        coordinates[1] = curr_traj[:, 1]
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
        depth_canvas = np.zeros(canvas_size, dtype=np.int16)
        for (col, row), d in zip(points.T, depth):
            depth_canvas[row, col] = d

        depth_canvas = torch.from_numpy(depth_canvas)

        inds = (depth_canvas > 0).nonzero()
        for ind in inds:
            y, x = ind
            x, y = x.item(), y.item()
            r = 5
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,255))
        
        img = Image.fromarray(img.astype('uint8'), 'RGB').convert('RGBA')
        final = Image.alpha_composite(img, overlay_img).convert('RGB')


        dir = f'/mnt/f/e2e/navsim_ours/debug/{output}'
        os.makedirs(dir, exist_ok=True)
        final.save(f'{dir}/{token}.png')


if __name__ == "__main__":
    main()
