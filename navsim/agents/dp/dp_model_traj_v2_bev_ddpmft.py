from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.dp.dp_model_traj_v1 import SimpleDiffusionTransformer
from navsim.agents.dp.dp_model_traj_v1_bev_ddpmft import DDPMSchedulerFt
from navsim.agents.dp.dp_model_traj_v2_bev import cumsum_traj, diff_traj
from navsim.agents.hydra_plus.hydra_backbone_bev import HydraBackboneBEV

HORIZON = 8
ACTION_DIM = 4
ACTION_DIM_ORI = 3


class DPHead_v2_ddpmft(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DPConfig = None
                 ):
        super().__init__()
        self.config = config
        self.noise_scheduler = DDPMSchedulerFt(
            num_train_timesteps=config.denoising_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type='epsilon'
        )
        img_num = 2 if config.use_back_view else 1

        self.transformer_dp = SimpleDiffusionTransformer(
            d_model, nhead, d_ffn, config.dp_layers,
            input_dim=ACTION_DIM * HORIZON,
            obs_len=config.img_vert_anchors * config.img_horz_anchors * img_num + 1,
        )

        self.reference_transformer = SimpleDiffusionTransformer(
            d_model, nhead, d_ffn, config.dp_layers,
            input_dim=ACTION_DIM * HORIZON,
            obs_len=config.img_vert_anchors * config.img_horz_anchors * img_num + 1,
        )
        self.reference_transformer.load_state_dict(self.transformer_dp.state_dict())
        self.reference_transformer.requires_grad_(False)
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps

    def update_reference_model(self):
        self.reference_transformer.load_state_dict(self.transformer_dp.state_dict())

    def forward(self, kv) -> Dict[str, torch.Tensor]:
        transformer = self.reference_transformer if self.config.is_rl_training else self.transformer_dp
        if self.config.reinforce:
            transformer = self.transformer_dp

        with torch.no_grad():
            B = kv.shape[0]
            result = {}
            N = self.config.num_proposals  # 从循环内部提取到此处

            condition = kv.repeat_interleave(N, dim=0)
            noise = torch.randn(
                size=(B * N, HORIZON, ACTION_DIM),
                dtype=condition.dtype,
                device=condition.device,
            )

            # 设置扩散步骤
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            latents = [noise]
            log_probs = []
            # 批量扩散过程 (保持时间步循环，但消除批次循环)
            for t in self.noise_scheduler.timesteps:
                model_output = transformer(
                    noise,
                    t,  # 确保时间步广播到所有样本
                    condition
                )
                noise_out, log_prob = self.noise_scheduler.step(
                    model_output, t, noise, stddev_clip=self.config.stdev_clip
                )
                noise = noise_out.prev_sample
                latents.append(noise)
                log_probs.append(log_prob)

            # 后处理并恢复形状 [B, NUM_PROPOSALS, HORIZON, ACTION_DIM]
            traj = cumsum_traj(noise)
            result['dp_pred'] = traj.view(B, N, HORIZON, ACTION_DIM_ORI)
            result['latents'] = latents
            result['log_probs'] = log_probs
            result['condition'] = condition
            result['timesteps'] = self.noise_scheduler.timesteps
            return result

    def get_dp_loss(self, kv, gt_trajectory):
        B = kv.shape[0]
        device = kv.device
        gt_trajectory = gt_trajectory.float()
        gt_trajectory = diff_traj(gt_trajectory)

        noise = torch.randn(gt_trajectory.shape, device=device, dtype=torch.float)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_dp_input = self.noise_scheduler.add_noise(
            gt_trajectory, noise, timesteps
        )

        # Predict the noise residual
        pred = self.transformer_dp(
            noisy_dp_input,
            timesteps,
            kv
        )
        return F.mse_loss(pred, noise) * self.config.dp_loss_weight


class DPModel_traj_v2_bev_ddpmft(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self._config = config
        assert config.backbone_type in ['vit', 'intern', 'vov', 'resnet', 'eva', 'moe', 'moe_ult32', 'swin']
        if config.backbone_type == 'eva':
            raise ValueError(f'{config.backbone_type} not supported')
        elif config.backbone_type == 'intern' or config.backbone_type == 'vov' or \
                config.backbone_type == 'swin' or config.backbone_type == 'vit' or config.backbone_type == 'resnet':
            self._backbone = HydraBackboneBEV(config)

        kv_len = self._backbone.bev_w * self._backbone.bev_h
        self._keyval_embedding = nn.Embedding(
            kv_len + 1, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Linear(self._backbone.img_feat_c, config.tf_d_model)
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        self._trajectory_head = DPHead_v2_ddpmft(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        if self._config.use_temporal_bev_kv:
            self.temporal_bev_fusion = nn.Conv2d(
                config.tf_d_model * 2,
                config.tf_d_model,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        camera_feature_back: torch.Tensor = features["camera_feature_back"]
        status_feature: torch.Tensor = features["status_feature"][0]

        batch_size = status_feature.shape[0]
        assert (camera_feature[-1].shape[0] == batch_size)

        camera_feature_curr = camera_feature[-1]
        if isinstance(camera_feature_back, list):
            camera_feature_back_curr = camera_feature_back[-1]
        else:
            camera_feature_back_curr = camera_feature_back
        img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_curr, camera_feature_back_curr)
        keyval = self.downscale_layer(bev_tokens)
        assert not self._config.use_temporal_bev_kv
        if self._config.use_temporal_bev_kv:
            with torch.no_grad():
                camera_feature_prev = camera_feature[-2]
                camera_feature_back_prev = camera_feature_back[-2]
                img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_prev, camera_feature_back_prev)
                keyval_prev = self.downscale_layer(bev_tokens)
            # grad for fusion layer
            C = keyval.shape[-1]
            keyval = self.temporal_bev_fusion(
                torch.cat([
                    keyval.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w),
                    keyval_prev.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w)
                ], 1)
            ).view(batch_size, C, -1).permute(0, 2, 1).contiguous()

        bev_semantic_map = self._bev_semantic_head(up_bev)
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([keyval, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval)

        output.update(trajectory)

        output['env_kv'] = keyval
        output['bev_semantic_map'] = bev_semantic_map

        return output
