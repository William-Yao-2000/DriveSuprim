import os
import pickle, shelve
from typing import Any, Union
import json

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.hydra_ssl.hydra_config_ssl import HydraConfigSSL
from navsim.agents.hydra_ssl.hydra_features_ssl import HydraSSLFeatureBuilder, HydraSSLTargetBuilder
from navsim.agents.hydra_ssl.ssl_meta_arch import SSLMetaArch
from navsim.agents.hydra_ssl.hydra_loss_fn_ssl import hydra_kd_imi_agent_loss_robust, hydra_kd_imi_agent_loss_single_stage
from navsim.agents.hydra_ssl.hydra_model_v_ssl import HydraModel

from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

DEVKIT_ROOT = os.getenv('NAVSIM_DEVKIT_ROOT')
TRAJ_PDM_ROOT = os.getenv('NAVSIM_TRAJPDM_ROOT')

from typing import Dict, List

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory


class HydraAgentSSL(AbstractAgent):
    def __init__(
            self,
            config: HydraConfigSSL,
            lr: float,
            checkpoint_path: str = None,
            pdm_split=None,
            metrics=None,
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        config.trajectory_pdm_weight = {
            'no_at_fault_collisions': 3.0,
            'drivable_area_compliance': 3.0,
            'time_to_collision_within_bound': 4.0,
            'ego_progress': 2.0,
            'driving_direction_compliance': 1.0,
            'lane_keeping': 2.0,
            'traffic_light_compliance': 3.0,
            'history_comfort': 1.0,
        }
        if config.lab.change_loss_weight:
            config.trajectory_pdm_weight = {
                'no_at_fault_collisions': 1.5,
                'drivable_area_compliance': 1.5,
                'time_to_collision_within_bound': 1.5,
                'ego_progress': 2.0,
                'driving_direction_compliance': 1.0,
                'lane_keeping': 2.0,
                'traffic_light_compliance': 1.0,
                'history_comfort': 1.0,
            }
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        self._config = config
        self._lr = lr
        self.metrics = metrics
        self._checkpoint_path = checkpoint_path
        teacher_model = HydraModel(config)
        student_model = HydraModel(config)
        self.model = SSLMetaArch(config, teacher_model, student_model)
        self.vocab_size = config.vocab_size
        self.backbone_wd = config.backbone_wd
        new_pkl_dir = f'vocab_score_full_{self.vocab_size}_navtrain'
        self.ensemble_aug = config.ego_perturb.ensemble_aug
        self.training = config.training

        if self.training:
            self.ori_vocab_pdm_score_full = pickle.load(
                open(f'{config.ori_vocab_pdm_score_full_path}', 'rb'))
            self.aug_vocab_pdm_score_dir = config.aug_vocab_pdm_score_dir
            
            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
            assert aug_data['param']['rot'] == config.ego_perturb.rotation.offline_aug_angle_boundary
            self.aug_info = aug_data['tokens']
        
        self.only_ori_input = config.only_ori_input
        self.n_rotation_crop = config.student_rotation_ensemble

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [HydraSSLTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraSSLFeatureBuilder(config=self._config)]

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        
        features, targets, tokens = batch
        kwargs = {'tokens': tokens}

        teacher_ori_features = dict()
        student_ori_features = dict()
        teacher_ori_features['camera_feature'] = features['ori_teacher']
        teacher_ori_features['status_feature'] = features['status_feature']

        student_feat_dict_lst = []
        
        student_ori_features['camera_feature'] = features['ori']
        student_ori_features['status_feature'] = features['status_feature']
        student_feat_dict_lst.append(student_ori_features)
        if not self.only_ori_input and self._config.training:
            for i in range(self.n_rotation_crop):
                student_feat_dict_lst.append(
                    {
                        'camera_feature': features['rotated'][i],
                        'status_feature': features['status_feature'],
                    }
                )

            if self._config.use_mask_loss:
                kwargs = {
                    'collated_masks': features['collated_masks'],
                    "mask_indices_list": features['mask_indices_list'],
                    "masks_weight": features['masks_weight'],
                    "upperbound": features['upperbound'],
                    "n_masked_patches": features['n_masked_patches'],
                }

        teacher_pred, student_preds, loss_dict = self.model(teacher_ori_features, student_feat_dict_lst, **kwargs)
        return teacher_pred, student_preds, loss_dict

    def forward_train(self, features, interpolated_traj):
        return self.vadv2_model(features, interpolated_traj)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: List[Dict[str, torch.Tensor]],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()
        # get the pdm score by tokens

        # ori
        ori_targets = { 'trajectory': targets['ori_trajectory'] }
        ori_predictions = predictions[0]
        scores = {}
        for k in self.metrics:
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                        .to(ori_predictions['trajectory'].device))
        ori_loss = hydra_kd_imi_agent_loss_robust(ori_targets, ori_predictions, self._config, scores)
        if self._config.only_ori_input:
            return { "ori": ori_loss }

        # aug
        _aug_vocab_pdm_score = {}
        for token in tokens:
            with open(os.path.join(self.aug_vocab_pdm_score_dir, f'{token}.pkl'), 'rb') as f:
                _aug_vocab_pdm_score[token] = pickle.load(f)
        aug_loss = []
        for idx in range(self._config.student_rotation_ensemble):
            aug_targets = { 'trajectory': targets['rotated_trajectories'][idx] }
            scores = {}
            for k in self.metrics:
                tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                            .to(predictions[idx+1]['trajectory'].device))
            aug_loss.append(hydra_kd_imi_agent_loss_robust(aug_targets, predictions[idx+1], self._config, scores))
        
        # Calculate average loss and loss dict
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss]))
        avg_aug_loss_dict = {}
        for key in aug_loss[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss]))
        return {
            "ori": ori_loss,
            "aug": (avg_aug_loss, avg_aug_loss_dict),
        }
    
    def compute_loss_soft_teacher(
            self,
            teacher_pred: Dict[str, torch.Tensor],
            student_pred: Dict[str, torch.Tensor],
            targets,
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        # get the pdm score by tokens

        sampled_timepoints = [5 * ii - 1 for ii in range(1, 9)]
        traj_diff = teacher_pred['trajectory'][:, sampled_timepoints] - targets['ori_trajectory']
        clamped_traj_diff = torch.clamp(traj_diff, min=-self._config.soft_label_diff_thresh, max=self._config.soft_label_diff_thresh)
        # Apply clamped adjustment to original trajectory
        revised_targets = { 'trajectory': targets['ori_trajectory'] + clamped_traj_diff }

        scores = {}
        revised_scores = {}
        for k in self.metrics:
            # if k == 'dd':
            #     continue
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = torch.from_numpy(np.concatenate(tmp, axis=0)).to(teacher_pred['trajectory'].device).float()
            # Calculate difference and clamp to max 0.2
            diff = teacher_pred[k].sigmoid() - scores[k]
            clamped_diff = torch.clamp(diff, min=-0.15, max=0.15)
            # Apply clamped adjustment to original scores
            revised_scores[k] = scores[k] + clamped_diff
        
        soft_loss = hydra_kd_imi_agent_loss_robust(revised_targets, student_pred, self._config, revised_scores)
        return soft_loss
    
    def compute_rotation_loss(
            self,
            teacher_pred: Dict[str, torch.Tensor],
            student_preds: List[Dict[str, torch.Tensor]],
            tokens
    ):
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        teacher_normed_cls = _get_norm_tensor(teacher_pred['cls_token_after_head'])  # (bs, 2)
        student_normed_cls_lst = [_get_norm_tensor(stu['cls_token_after_head']) for stu in student_preds[1:1+self._config.student_rotation_ensemble]]  # list of (bs, 2)
        gt_augs = [self.aug_info[token] for token in tokens]
        gt_angles = torch.tensor([[aug['rot'] for aug in gt_aug] for gt_aug in gt_augs], device=teacher_normed_cls.device)  # (bs, n_aug)
        gt_angles = gt_angles * np.pi / 180.0  # convert to radians

        rot_loss = 0
        for i, student_cls in enumerate(student_normed_cls_lst):
            # Calculate sin and cos between teacher and student
            sin_val = teacher_normed_cls[:,0] * student_cls[:,1] - teacher_normed_cls[:,1] * student_cls[:,0]  # (bs)
            cos_val = teacher_normed_cls[:,0] * student_cls[:,0] + teacher_normed_cls[:,1] * student_cls[:,1]  # (bs)
            
            # Compare with ground truth angles
            gt_sin = torch.sin(gt_angles[:, i])  # (bs)
            gt_cos = torch.cos(gt_angles[:, i])  # (bs)
            
            # Calculate L1 loss
            rot_loss += torch.mean(torch.abs(sin_val - gt_sin) * 5.0 + torch.abs(cos_val - gt_cos))

        rot_loss = rot_loss / len(student_normed_cls_lst)
        return rot_loss
    
    def compute_loss_multi_stage(
        self,
        features,
        targets: Dict[str, torch.Tensor],
        predictions: List[Dict[str, torch.Tensor]],
        tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()

        trajectory_vocab = predictions[0]['trajectory_vocab']

        result_dict = dict()
        # ori
        ori_loss_lst = []
        ori_predictions = predictions[0]['refinement']
        num_stage = len(ori_predictions)
        for i in range(num_stage):
            pred_i = ori_predictions[i]
            selected_indices_i = pred_i['indices_absolute']
            scores = {}
            for k in self.metrics:
                tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
                full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(selected_indices_i.device)  # [bs, vocab_size]
                # Extract scores based on selected indices [bs, topk_stage_i]
                batch_size, topk = selected_indices_i.shape
                batch_indices = torch.arange(batch_size, device=selected_indices_i.device).unsqueeze(1).expand(-1, topk)
                scores[k] = full_scores[batch_indices, selected_indices_i]  # [bs, topk_stage_i]
            _kwargs = {}
            if self._config.lab.use_imi_learning_in_refinement:
                _kwargs['targets'] = { 'trajectory': targets['ori_trajectory'] }
                pred_i['trajectory_vocab'] = trajectory_vocab
            ori_loss_i = hydra_kd_imi_agent_loss_single_stage(pred_i, self._config, scores, **_kwargs)
            ori_loss_lst.append(ori_loss_i)
        total_ori_loss = sum([loss_tup[0] for loss_tup in ori_loss_lst])
        total_ori_loss_dict = {}
        for i, loss_tup in enumerate(ori_loss_lst):
            loss_dict = loss_tup[1]
            for _key, _value in loss_dict.items():
                total_ori_loss_dict[f"stage_{i+2}_{_key}"] = _value
        result_dict['ori'] = (total_ori_loss, total_ori_loss_dict)
        if self._config.only_ori_input:
            return result_dict

        # aug
        _aug_vocab_pdm_score = {}
        for token in tokens:
            with open(os.path.join(self.aug_vocab_pdm_score_dir, f'{token}.pkl'), 'rb') as f:
                _aug_vocab_pdm_score[token] = pickle.load(f)
        aug_loss_all_mode_lst = []
        for idx in range(self._config.student_rotation_ensemble):
            aug_loss_lst = []
            aug_idx_predictions = predictions[idx+1]['refinement']
            for i in range(num_stage):
                aug_idx_pred_i = aug_idx_predictions[i]
                aug_idx_selected_indices_i = aug_idx_pred_i['indices_absolute']
                scores = {}
                for k in self.metrics:
                    tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                    full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(aug_idx_selected_indices_i.device)
                    batch_size, topk = aug_idx_selected_indices_i.shape
                    batch_indices = torch.arange(batch_size, device=aug_idx_selected_indices_i.device).unsqueeze(1).expand(-1, topk)
                    scores[k] = full_scores[batch_indices, aug_idx_selected_indices_i]
                _kwargs_idx = {}
                if self._config.lab.use_imi_learning_in_refinement:
                    _kwargs_idx['targets'] = { 'trajectory': targets['rotated_trajectories'][idx] }
                    aug_idx_pred_i['trajectory_vocab'] = trajectory_vocab
                aug_loss_lst.append(hydra_kd_imi_agent_loss_single_stage(aug_idx_pred_i, self._config, scores, **_kwargs_idx))
            aug_loss_single_mode = sum([loss_tup[0] for loss_tup in aug_loss_lst])
            aug_loss_single_mode_dict = {}
            for i, loss_tup in enumerate(aug_loss_lst):
                loss_dict = loss_tup[1]
                for _key, _value in loss_dict.items():
                    aug_loss_single_mode_dict[f"stage_{i+2}_{_key}"] = _value
            aug_loss_all_mode_lst.append((aug_loss_single_mode, aug_loss_single_mode_dict))
        
        # Calculate average loss and loss dict
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss_all_mode_lst]))
        avg_aug_loss_dict = {}
        for key in aug_loss_all_mode_lst[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss_all_mode_lst]))
        result_dict['aug'] = (avg_aug_loss, avg_aug_loss_dict)

        return result_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.student.model.named_parameters()))
        default_params = list(filter(lambda kv: backbone_params_name not in kv[0], self.model.student.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]
        return torch.optim.Adam(params_lr_dict, lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            # TransfuserCallback(self._config),
            ModelCheckpoint(
                save_top_k=30,
                monitor="val/loss-ori",
                mode="min",
                dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
                filename="{epoch:02d}-{step:04d}",
            )
        ]
    
def _get_norm_tensor(t):
    norms = torch.norm(t, p=2, dim=1, keepdim=True)
    return t / norms
