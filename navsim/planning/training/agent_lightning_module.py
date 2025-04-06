from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_agent import DPAgent
from navsim.agents.hydra_plus.hydra_features import state2traj
from navsim.agents.hydra_plus.hydra_plus_agent import HydraPlusAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.convert import absolute_to_relative_poses
import numpy as np

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.simulator = PDMSimulator(
            TrajectorySampling(num_poses=40, interval_length=0.1)
        )
        self.v_params = get_pacifica_parameters()

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens = batch

        prediction = self.agent.forward(features)

        if isinstance(self.agent, TransfuserAgent):
            loss, loss_dict = self.agent.compute_loss(features, targets, prediction)
        else:
            loss, loss_dict = self.agent.compute_loss(features, targets, prediction, tokens)

        for k, v in loss_dict.items():
            self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

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
        if isinstance(self.agent, HydraPlusAgent):
            return self.predict_step_hydra(batch, batch_idx)
        elif isinstance(self.agent, DPAgent):
            return self.predict_step_dp(batch, batch_idx)
        else:
            raise ValueError('unsupported agent')

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

        result = {}
        for (proposals, token) in zip(all_trajs, tokens):
            # todo use hydra to sample
            pose = proposals[0]
            result[token] = {
                'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
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
        # todo inference
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"].cpu().numpy()

            imis = predictions["imi"].softmax(-1).log().cpu().numpy()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].sigmoid().log().cpu().numpy()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].sigmoid().log().cpu().numpy()
            time_to_collision_within_bound_all = predictions[
                "time_to_collision_within_bound"].sigmoid().log().cpu().numpy()
            ego_progress_all = predictions["ego_progress"].sigmoid().log().cpu().numpy()
            driving_direction_compliance_all = predictions["driving_direction_compliance"].sigmoid().log().cpu().numpy()
            lane_keeping_all = predictions["lane_keeping"].sigmoid().log().cpu().numpy()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].sigmoid().log().cpu().numpy()
            # history_comfort_all = predictions["history_comfort"].sigmoid().log().cpu().numpy()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        result = {}
        for (pose,
             imi,
             no_at_fault_collisions,
             drivable_area_compliance,
             time_to_collision_within_bound,
             ego_progress,
             driving_direction_compliance,
             lane_keeping,
             traffic_light_compliance,
             # history_comfort,
             token) in \
                zip(poses,
                    imis,
                    no_at_fault_collisions_all,
                    drivable_area_compliance_all,
                    time_to_collision_within_bound_all,
                    ego_progress_all,
                    driving_direction_compliance_all,
                    lane_keeping_all,
                    traffic_light_compliance_all,
                    # history_comfort_all,
                    tokens):
            result[token] = {
                'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                'imi': imi,
                'no_at_fault_collisions': no_at_fault_collisions,
                'drivable_area_compliance': drivable_area_compliance,
                'time_to_collision_within_bound': time_to_collision_within_bound,
                'ego_progress': ego_progress,
                'driving_direction_compliance': driving_direction_compliance,
                'lane_keeping': lane_keeping,
                'traffic_light_compliance': traffic_light_compliance,
                # 'history_comfort': history_comfort
            }
        return result
