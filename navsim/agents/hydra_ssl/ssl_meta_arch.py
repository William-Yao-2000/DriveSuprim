# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import os

import torch
from torch import nn

from navsim.agents.hydra_ssl.hydra_config_ssl import HydraConfigSSL
from navsim.agents.hydra_ssl.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from navsim.agents.hydra_ssl.layers import DINOHead
from navsim.agents.hydra_ssl.utils.fsdp import get_fsdp_modules


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg: HydraConfigSSL, teacher_model, student_model):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_model_dict["model"] = teacher_model
        teacher_model_dict["model"] = student_model
        embed_dim = teacher_model._backbone.img_feat_c
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes  # 65536

        self.do_dino = cfg.dino.loss_weight > 0
        # self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.use_mask_loss
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,  # 65536
                hidden_dim=cfg.dino.head_hidden_dim,  # 2048
                bottleneck_dim=cfg.dino.head_bottleneck_dim,  # 256
                nlayers=cfg.dino.head_nlayers,
            )

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

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
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            teacher_output_dict = self.teacher.model(teacher_ori_features)
            """
            teacher_backbone_output_dict:
            {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }
            """
            # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            #     import pdb; pdb.set_trace()
            teacher_cls_tokens = teacher_output_dict["class_token"]  # [bs * n_global_crops, c]

            teacher_cls_token_after_head = self.teacher.dino_head(teacher_cls_tokens)
            teacher_output_dict['cls_token_after_head'] = teacher_cls_token_after_head

            masked_teacher_ibot_softmaxed_centered = None
            if self.do_ibot and self.cfg.training:
                upperbound = kwargs['upperbound']
                mask_indices_list = kwargs['mask_indices_list']
                n_masked_patches = kwargs['n_masked_patches']
                teacher_temp = 0.07  # temperature

                ibot_teacher_patch_tokens = teacher_output_dict["patch_token"].flatten(2).permute(0, 2, 1)
                _dim = ibot_teacher_patch_tokens.shape[-1]
                
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]

                masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)  # [1, nmp, c_dino]
                masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                    masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                )  # [1, nmp, c_dino]
                masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)  # [nmp, c_dino]
                self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            return teacher_output_dict, masked_teacher_ibot_softmaxed_centered

        if self.cfg.training:
            teacher_pred, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        else:
            if self.cfg.inference.model == "teacher":
                teacher_pred = self.teacher.model(teacher_ori_features, **kwargs)
            else:
                teacher_pred = self.student.model(teacher_ori_features, **kwargs)
        # reshard_fsdp_model(self.teacher)
        if not self.cfg.training:
            return teacher_pred, [], {}
        
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        
        if self.cfg.lab.optimize_prev_frame_traj_for_ec:
            teacher_pred = {'cur': teacher_pred}
            teacher_prev_feat = {
                'camera_feature': [teacher_ori_features['camera_feature'][-2],],
                'status_feature': [teacher_ori_features['status_feature'][1],],
            }
            prev_pred = self.teacher.model(teacher_prev_feat, **kwargs)
            teacher_pred['prev'] = prev_pred


        loss_dict = {}

        student_preds = self.student.model.forward_features_list(
            student_feat_dict_lst,
            mask_list=[kwargs.get('collated_masks', None)] + [None] * len(student_feat_dict_lst[1:]),
        )

        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()

        inputs_for_student_head_lst = [pred["class_token"] for pred in student_preds]

        for i in range(len(student_preds)):
            student_preds[i]['cls_token_after_head'] = self.student.dino_head(inputs_for_student_head_lst[i])
        
        if self.do_ibot:
            masks = kwargs['collated_masks']
            upperbound = kwargs['upperbound']
            mask_indices_list = kwargs['mask_indices_list']
            n_masked_patches = kwargs['n_masked_patches']
            masks_weight = kwargs['masks_weight']

            student_masked_output_dict = student_preds[0]
            _dim = student_masked_output_dict["class_token"].shape[-1]
            ibot_student_patch_tokens = student_masked_output_dict["patch_token"].flatten(2).permute(0, 2, 1)
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)  # [ub, c]
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                :n_masked_patches
            ]

            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,  # [nmp, c_dino]
                    masked_teacher_ibot_softmaxed_centered,  # [nmp, c_dino]
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
            )

            # store for display
            loss_dict["loss_ibot"] = ibot_patch_loss * self.cfg.ibot.loss_weight

        return teacher_pred, student_preds, loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = (
                self.teacher.dino_head._streams
            ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()
        with torch.no_grad():
            for k in self.student.keys():
                for stu_params, tea_params in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    tea_params.data.mul_(m).add_(stu_params.data, alpha=1 - m)
                if self.cfg.backbone_type in ('resnet34', 'resnet50') or self.cfg.lab.update_buffer_in_ema:
                    # update buffers (e.g., running_mean/var in BatchNorm)
                    for stu_buf, tea_buf in zip(self.student[k].buffers(), self.teacher[k].buffers()):
                        tea_buf.data.copy_(stu_buf.data)
                
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        #     for k, v in self.teacher.model.state_dict().items():
        #         v2 = self.student.model.state_dict().get(k)
        #         if v2 is None:
        #             print(f"{k} only in teacher")
        #         elif not torch.equal(v, v2):
        #             print(f"{k} differs")
        #     import pdb; pdb.set_trace()

    def train(self, mode=True):
        if mode:
            super().train()
            self.teacher.eval()
        else:
            self.teacher.eval()
            self.student.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
