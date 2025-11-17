import cv2
from enum import IntEnum
import json, os
import numpy as np
import numpy.typing as npt
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon, LineString
from typing import Any, Dict, List, Tuple, Union

import torch
from torchvision import transforms

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.data.transforms import GaussianBlur
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.metric_caching.metric_cache_processor_aug_train import ego_state_augmentation


class DriveSuprimFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: DriveSuprimConfig):
        self._config = config
        self.training = config.training

        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()

        if self.training:
            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
            assert aug_data['param']['rot'] == config.ego_perturb.offline_aug_angle_boundary
            self.aug_info = aug_data['tokens']

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        self.teacher_ori_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.student_ori_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                color_jittering,
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=0.5, p=0.2),
            ]
        )
        self.student_rotated_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                color_jittering,
                GaussianBlur(p=0.5),
            ]
        )

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "drivesuprim_feature"

    def compute_features(self, agent_input: AgentInput, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()

        features = {}
        
        initial_token = scene.scene_metadata.initial_token
        if not self._config.only_ori_input and self._config.training:
            n_rotated = self._config.ego_perturb.n_student_rotation_ensemble
        else:
            n_rotated = 0
        features.update(self._get_camera_feature(agent_input, initial_token, rotation_num=n_rotated))

        ego_status_list = []
        for i in range(self._config.seq_len):
            idx = -(i + 1)
            ego_status = torch.concatenate(
                [
                torch.tensor(agent_input.ego_statuses[idx].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[idx].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[idx].ego_acceleration, dtype=torch.float32)
                ],
            )
            ego_status_list.append(ego_status)

        features["status_feature"] = ego_status_list  # [seq_len, 8]

        return features

    def _get_camera_feature(self, agent_input: AgentInput, initial_token: str, rotation_num=3) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Extract teacher and student camera input from AgentInput
        :param 
          agent_input: input dataclass
          initial_token: scene token, used to get the specific rotation angle of augmentation
          rotation_num: number of rotation angle
        """
        res = dict()

        seq_len = self._config.seq_len
        cameras = agent_input.cameras[-seq_len:]  # List[Cameras]
        assert(len(cameras) == seq_len)

        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        
        # Teacher input
        res['ori_teacher'] = []

        # Student input
        res['ori'] = []
        res['rotated'] = [[] for _ in range(rotation_num)]

        for camera in cameras:
            image = camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                n_camera = self._config.n_camera

                # Crop to ensure 4:1 aspect ratio
                l1 = camera.cam_l1.image[28:-28]
                l0 = camera.cam_l0.image[28:-28, 416:-416]
                f0 = camera.cam_f0.image[28:-28]
                r0 = camera.cam_r0.image[28:-28, 416:-416]
                r1 = camera.cam_r1.image[28:-28]
                if n_camera >= 5:
                    l2 = camera.cam_l2.image[28:-28, :-1100]
                    r2 = camera.cam_r2.image[28:-28, 1100:]
                    b0_left = camera.cam_b0.image[28:-28, :1080]
                    b0_right = camera.cam_b0.image[28:-28, -1080:]
                
                if n_camera == 1:
                    ori_image = f0
                elif n_camera == 3:
                    ori_image = np.concatenate([l0, f0, r0], axis=1)
                elif n_camera == 5:
                    ori_image = np.concatenate([l1, l0, f0, r0, r1], axis=1)
                else:
                    raise NotImplementedError(f"n_camera={n_camera} is not supported")
                
                _ori_image = cv2.resize(ori_image, (self._config.camera_width, self._config.camera_height))
                res['ori_teacher'].append(self.teacher_ori_augmentation(_ori_image))
                res['ori'].append(self.student_ori_augmentation(_ori_image))
                
                if n_camera < 5:
                    stitched_image = np.concatenate([l1, l0, f0, r0, r1], axis=1)
                else:
                    stitched_image = np.concatenate([b0_left, l2, l1, l0, f0, r0, r1, r2, b0_right], axis=1)
                
                img_3cam_w = l0.shape[1] + f0.shape[1] + r0.shape[1]
                l1_w = l1.shape[1]
                r1_w = r1.shape[1]
                whole_w = stitched_image.shape[1]
                half_view_w = img_3cam_w + l1_w // 2 + r1_w // 2
                if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                    debug_dir = 'debug_viz'
                    os.makedirs(debug_dir, exist_ok=True)
                    _s_img = Image.fromarray(stitched_image)
                    _s_img.save(f'{debug_dir}/stitched_image.jpg')

                if n_camera == 1:
                    img_w = f0.shape[1]
                elif n_camera == 3:
                    img_w = img_3cam_w
                elif n_camera == 5:
                    img_w = img_3cam_w + l1_w + r1_w
                else:
                    raise NotImplementedError(f"n_camera={n_camera} is not supported")

                for i in range(rotation_num):
                    _ego_rotation_angle_degree = self.aug_info[initial_token][i]['rot']
                    offset_w = int(half_view_w / 180 * _ego_rotation_angle_degree)
                    
                    rotated_image = stitched_image[:, int(whole_w/2-offset_w-img_w/2):int(whole_w/2-offset_w+img_w/2)]
                    resized_image = cv2.resize(rotated_image, (self._config.camera_width, self._config.camera_height))
                    tensor_image = self.student_rotated_augmentation(resized_image)  # [3, h, w]

                    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                        _img = transforms.ToPILImage()(tensor_image)
                        _img.save(f'{debug_dir}/output_tensor_image_{i}.jpg')
                    res['rotated'][i].append(tensor_image)
                
        return res


class DriveSuprimTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: DriveSuprimConfig):
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        
        self._config = config
        self.v_params = get_pacifica_parameters()
        self.training = config.training

        if self.training:
            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
                assert aug_data['param']['rot'] == config.ego_perturb.offline_aug_angle_boundary
                self.aug_info = aug_data['tokens']

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        initial_token = scene.scene_metadata.initial_token
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=int(4 / 0.5)
        )  # [8, 3] (local pose, do not include present pose)

        if not self._config.only_ori_input and self._config.training:
            n_rotated = self._config.ego_perturb.n_student_rotation_ensemble
            _rotations = [self.aug_info[initial_token][i]['rot'] for i in range(n_rotated)]
            _reverse_rotations = [-_rot / 180.0 * np.pi for _rot in _rotations]  # Convert degree to rad
        else:
            _rotations = [0]
            _reverse_rotations = [0]

        # Original trajectory
        ori_trajectory = torch.tensor(future_traj.poses)  # [num_poses, 3]

        # Apply rotations to get multiple trajectories
        rotated_trajectories = []
        for _reverse_rotation in _reverse_rotations:
            if self.training and abs(_reverse_rotation) > 1e-5:
                # Rotate each trajectory point around origin (ego vehicle)
                rotated_poses = []
                for pose in future_traj.poses:
                    # Rotate x,y coordinates
                    rotated_xy = np_vector2_aug(pose[:2], _reverse_rotation)
                    # Adjust heading and normalize to [-pi, pi]
                    rotated_heading = (pose[2] + _reverse_rotation + np.pi) % (2*np.pi) - np.pi
                    rotated_poses.append(np.array([rotated_xy[0], rotated_xy[1], rotated_heading]))
                rotated_poses = np.array(rotated_poses)
                rotated_trajectories.append(torch.tensor(rotated_poses))  # [num_poses, 3]
            else:
                rotated_trajectories.append(ori_trajectory)  # Use original trajectory when no rotation
                    
        ret = {
            "ori_trajectory": ori_trajectory,
            "rotated_trajectories": rotated_trajectories,
        }
        
        return ret


def np_vector2_aug(arr, angle):
    # angle: rad, positive means turning left
    _sin, _cos = np.sin(angle), np.cos(angle)
    x_rotated = arr[0] * _cos - arr[1] * _sin
    y_rotated = arr[0] * _sin + arr[1] * _cos
    return np.array([x_rotated, y_rotated])
