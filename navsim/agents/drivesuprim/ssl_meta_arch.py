# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import os

import torch
from torch import nn

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg: DriveSuprimConfig, teacher_model, student_model):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_model_dict["model"] = teacher_model
        teacher_model_dict["model"] = student_model

        self.do_dino = False
        self.do_ibot = False

        logger.info("OPTIONS -- DINO -- not using DINO")
        logger.info("OPTIONS -- iBOT -- not using iBOT")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        logger.info(f"Student and Teacher are built: they are both {type(self.student.model)} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward(self, teacher_ori_features, student_feat_dict_lst, **kwargs):

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            teacher_output_dict = self.teacher.model(teacher_ori_features)

            return teacher_output_dict

        if self.cfg.training:
            teacher_pred = get_teacher_output()
        else:
            if self.cfg.inference.model == "teacher":
                teacher_pred = self.teacher.model(teacher_ori_features, **kwargs)
            else:
                teacher_pred = self.student.model(teacher_ori_features, **kwargs)
        
        if not self.cfg.training:
            return teacher_pred, []
        
        student_preds = self.student.model.forward_features_list(student_feat_dict_lst)

        return teacher_pred, student_preds

    def update_teacher(self, m):

        with torch.no_grad():
            for k in self.student.keys():
                for stu_params, tea_params in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    tea_params.data.mul_(m).add_(stu_params.data, alpha=1 - m)
                if self.cfg.backbone_type in ('resnet34', 'resnet50') or self.cfg.update_buffer_in_ema:
                    # update buffers (e.g., running_mean/var in BatchNorm)
                    for stu_buf, tea_buf in zip(self.student[k].buffers(), self.teacher[k].buffers()):
                        tea_buf.data.copy_(stu_buf.data)

    def train(self, mode=True):
        if mode:
            super().train()
            self.teacher.eval()
        else:
            self.teacher.eval()
            self.student.eval()

