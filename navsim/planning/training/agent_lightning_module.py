from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Trajectory


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

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
        self.total_predictions += 1
        return self.predict_step_hydra(batch, batch_idx)

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

        return {token: {
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
