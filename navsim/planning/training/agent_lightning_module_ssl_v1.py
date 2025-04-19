import os, time
import pickle
from typing import Dict, Tuple, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import Tensor
import torch.nn.functional as F

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_agent import DPAgent
from navsim.agents.hydra_plus.hydra_features import state2traj
from navsim.agents.hydra_plus.hydra_plus_agent import HydraPlusAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.hydra_ssl_v1.hydra_config_ssl import HydraConfigSSL
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


class AgentLightningModuleSSL(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self,
                 cfg: HydraConfigSSL,
                 agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()

        self._cfg = cfg
        self.agent = agent
        self.simulator = PDMSimulator(
            TrajectorySampling(num_poses=40, interval_length=0.1)
        )
        self.v_params = get_pacifica_parameters()
        # self.hydra_preds = pickle.load(open(f'{os.getenv("NAVSIM_EXP_ROOT")}/hydra_plus_v2ep/epoch19.pkl', 'rb'))
        # self.vocab = torch.from_numpy(
        #     np.load(f'{os.getenv("NAVSIM_DEVKIT_ROOT")}/traj_final/test_16384_rearaxle_kmeans.npy')
        # )

        self.only_ori_input = cfg.only_ori_input
        self.n_rotation_crop = cfg.student_rotation_ensemble

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens = batch
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()

        teacher_pred, student_preds, loss_dict = self.agent.forward(batch)

        loss_student = self.agent.compute_loss(features, targets, student_preds, tokens)

        ori_loss = loss_student['ori']
        for k, v in ori_loss[1].items():
            self.log(f"{logging_prefix}/{k}-ori", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss-ori", ori_loss[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        loss = ori_loss[0]

        if not self.only_ori_input:
            aug_loss = loss_student['aug']
            for k, v in aug_loss[1].items():
                self.log(f"{logging_prefix}/{k}-aug", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss-aug", aug_loss[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            loss = loss + aug_loss[0]
        
            if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                import pdb; pdb.set_trace()

            loss_soft_teacher = self.agent.compute_loss_soft_teacher(teacher_pred, student_preds[0], targets, tokens)
            for k, v in loss_soft_teacher[1].items():
                self.log(f"{logging_prefix}/{k}-soft", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss-soft", loss_soft_teacher[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            loss = loss + loss_soft_teacher[0]

            if self._cfg.use_rotation_loss:
                loss_rotation = self.agent.compute_rotation_loss(teacher_pred, student_preds, tokens)
                self.log(f"{logging_prefix}/loss-rotation", loss_rotation, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                loss += loss_rotation
            else:
                loss += student_preds[0]['cls_token_after_head'].sum() * 0.0

            if self._cfg.use_mask_loss:
                loss_ibot = loss_dict['loss_ibot']
                self.log(f"{logging_prefix}/loss-ibot", loss_ibot, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                loss += loss_ibot

            if self._cfg.refinement.use_multi_stage:
                loss_refinement = self.agent.compute_loss_multi_stage(features, targets, student_preds, tokens)
                
                loss_refinement_ori = loss_refinement['ori']
                for k, v in loss_refinement_ori[1].items():
                    self.log(f"{logging_prefix}/{k}-ori", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/loss-refinement_ori", loss_refinement_ori[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                loss = loss + loss_refinement_ori[0]
                loss_refinement_aug = loss_refinement['aug']
                for k, v in loss_refinement_aug[1].items():
                    self.log(f"{logging_prefix}/{k}-aug", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/loss-refinement_aug", loss_refinement_aug[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                loss = loss + loss_refinement_aug[0]

        
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        

        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")
    
    def on_train_start(self):
        self.agent.model.train()


    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()
        if epoch < 3:
            m = 0.992 + epoch * 0.002
        else:
            m = 0.998
        self.agent.model.update_teacher(m)
    

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def predict_step(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        return self.predict_step_hydra(batch, batch_idx)

    def predict_step_dp(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            # [B, PROPOSAL, HORIZON, 2]
            controls = predictions["dp_pred"].cpu().numpy()

        all_trajs = []
        for batch_idx, command_states in enumerate(controls):
            ego_state = EgoState.build_from_rear_axle(
                StateSE2(*features['ego_pose'].cpu().numpy()[batch_idx]),
                tire_steering_angle=0.0,
                vehicle_parameters=self.v_params,
                time_point=TimePoint(0),
                rear_axle_velocity_2d=StateVector2D(
                    *features['ego_velocity'].cpu().numpy()[batch_idx]
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    *features['ego_acceleration'].cpu().numpy()[batch_idx]
                ),
            )
            traj_proposals = []
            for command_state in command_states:
                recovered_state = self.simulator.command_states2waypoints(
                    command_state, ego_state
                )
                recovered_traj = state2traj(recovered_state.squeeze(0))
                traj_proposals.append(recovered_traj)
            all_trajs.append(np.array(traj_proposals))

        # [B, PROPOSALS, HORIZON, 3]
        all_trajs = np.array(all_trajs)

        if all_trajs.shape[2] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        device = features['ego_pose'].device
        result = {}
        for (proposals, control, token) in zip(all_trajs, controls, tokens):
            # todo use hydra to sample
            # pose = proposals[0]
            hydra_result = self.hydra_preds[token]

            proposals_ = torch.from_numpy(proposals).to(device)
            vocab_ = self.vocab.to(device)
            dist = ((proposals_.unsqueeze(1) - vocab_.unsqueeze(0)) ** 2).sum((-1, -2))
            dist_argmin = dist.argmin(1).cpu().numpy()

            scores = (
                    0.5 * hydra_result['no_at_fault_collisions'][dist_argmin] +
                    0.5 * hydra_result['traffic_light_compliance'][dist_argmin] +
                    0.5 * hydra_result['drivable_area_compliance'][dist_argmin] +
                    0.5 * hydra_result['driving_direction_compliance'][dist_argmin] +
                    8.0 * np.log(
                5.0 * np.exp(hydra_result['time_to_collision_within_bound'][dist_argmin]) +
                5.0 * np.exp(hydra_result['ego_progress'][dist_argmin]) +
                2.0 * np.exp(hydra_result['lane_keeping'][dist_argmin])
            )
            )
            # prevent from going backwards:
            # 1. the overall trajectory goes backwards for 2m
            # 2. the trajectory starts from x < -0.5m
            backward_mask = (proposals_[..., 0] < -2.0).any(1).logical_and(proposals_[..., 0, 0] < -0.5)
            scores[backward_mask.cpu().numpy()] -= 100.0

            pose = proposals[scores.argmax(0)]
            result[token] = {
                'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                'proposals': proposals,
                'controls': control
            }

        # debug
        # min_dist = ((((targets['interpolated_traj'][:, None] - torch.from_numpy(all_trajs).to(targets['interpolated_traj'].device))[..., :2]) ** 2)
        #  .sum((-1,-2))
        #  .min(1))

        return result

    def predict_step_hydra(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions, _, _ = self.agent.forward(batch)
            if self._cfg.refinement.use_multi_stage:
                poses = predictions['final_traj'].cpu().numpy()
            else:
                poses = predictions["trajectory"].cpu().numpy()
            
            if 'trajectory_pre' in predictions:
                poses_pre = predictions['trajectory_pre'].cpu().numpy()
            else:
                poses_pre = poses

            imis = predictions["imi"].softmax(-1).log().cpu().numpy()
            nocs = predictions["noc"].log().cpu().numpy()
            das = predictions["da"].log().cpu().numpy()
            ttcs = predictions["ttc"].log().cpu().numpy()
            if 'comfort' in predictions:
                comforts = predictions["comfort"].log().cpu().numpy()
            else:
                comforts = np.zeros(len(tokens))
            if 'lk' in predictions:
                lks = predictions["lk"].log().cpu().numpy()
            else:
                lks = np.zeros(len(tokens))
            if 'tl' in predictions:
                tls = predictions["tl"].log().cpu().numpy()
            else:
                tls = np.zeros(len(tokens))
            if 'dr' in predictions:
                drs = predictions["dr"].log().cpu().numpy()
            else:
                drs = np.zeros(len(tokens))
            if 'progress' in predictions:
                progresses = predictions["progress"].log().cpu().numpy()
            else:
                progresses = [None for _ in range(len(tokens))]

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        ret_dict = {token: {
            'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
            'trajectory_pre': Trajectory(pose_pre, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
            'imi': imi,
            'noc': noc,
            'da': da,
            'ttc': ttc,
            'comfort': comfort,
            'progress': progress,
            'lk': lk,
            'tl': tl,
            'dr': dr,
        } for pose, pose_pre, imi, noc, da, ttc, comfort, progress, lk, tl, dr, token in
            zip(poses, poses_pre, imis, nocs, das, ttcs, comforts, progresses,
                lks, tls, drs, tokens)}
        return ret_dict
