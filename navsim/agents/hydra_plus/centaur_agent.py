import os
from typing import Dict

import torch

from navsim.agents.hydra_ssl.hydra_agent_ssl import HydraAgentSSL
from navsim.agents.hydra_ssl.hydra_config_ssl import HydraConfigSSL

DEVKIT_ROOT = os.getenv('NAVSIM_DEVKIT_ROOT')
TRAJ_PDM_ROOT = os.getenv('NAVSIM_TRAJPDM_ROOT')


class CentaurAgent(HydraAgentSSL):
    def __init__(
            self,
            config: HydraConfigSSL,
            lr: float,
            checkpoint_path: str = None,
            pdm_split=None,
            metrics=None,
    ):
        super().__init__(
            config=config,
            lr=lr,
            checkpoint_path=checkpoint_path,
            pdm_split=pdm_split,
            metrics=metrics
        )
        # todo sample probs
        self.sample_probs = None

    def forward(self, batch) -> Dict[str, torch.Tensor]:
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

        _, _, loss_dict = self.model(teacher_ori_features, student_feat_dict_lst, **kwargs)

        # todo centaur ttt
        loss = loss_dict['something']
        loss.backward()
        optimzer = None
        optimzer.step()

        # forward again
        teacher_pred, student_preds, loss_dict = self.model(teacher_ori_features, student_feat_dict_lst, **kwargs)

        # reset
        self.initialize()
        return teacher_pred, student_preds, loss_dict
