import os
from typing import Any, Union

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.dp.dp_model import DPModel
from navsim.agents.hydra_plus.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

DEVKIT_ROOT = os.getenv('NAVSIM_DEVKIT_ROOT')
TRAJ_PDM_ROOT = os.getenv('NAVSIM_TRAJPDM_ROOT')

from typing import List

import pytorch_lightning as pl
from navsim.agents.abstract_agent import AbstractAgent
from typing import Dict

import torch
import torch.nn.functional as F

from navsim.agents.hydra_plus.hydra_config import HydraConfig


def dp_loss(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor],
        config: HydraConfig, traj_head
):
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    dp_loss = traj_head.get_dp_loss(predictions['env_kv'], target_traj.float())
    loss = (
        dp_loss
    )
    return loss, {
        'dp_loss': dp_loss
    }


class DPAgent(AbstractAgent):
    def __init__(
            self,
            config: DPConfig,
            lr: float,
            checkpoint_path: str = None,
            pdm_gt_path=None,
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.model = DPModel(config)
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

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
        return [HydraTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def forward_train(self, features, interpolated_traj):
        return self.model(features, interpolated_traj)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # get the pdm score by tokens
        return dp_loss(targets, predictions, self._config, self.model._trajectory_head)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.named_parameters()))
        default_params = list(filter(lambda kv: backbone_params_name not in kv[0], self.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]
        if self.scheduler == 'default':
            return torch.optim.Adam(params_lr_dict, lr=self._lr)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.001,
                    total_steps=20 * 196
                )
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            # TransfuserCallback(self._config),
            ModelCheckpoint(
                save_top_k=30,
                monitor="val/loss_epoch",
                mode="min",
                dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
                filename="{epoch:02d}-{step:04d}",
            )
        ]
