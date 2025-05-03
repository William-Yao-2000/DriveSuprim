from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.dp.dp_model import SinusoidalPosEmb, TemporalAttention
from navsim.agents.hydra_plus.hydra_backbone import HydraBackbone

scale_x_diff = 7.5
scale_ys = [0.689, 1.95, 3.78, 6.14, 9.33, 13.33, 17.64, 22.33]

HORIZON = 8
ACTION_DIM = 4
ACTION_DIM_ORI = 3


def diff_traj(traj):
    B, L, _ = traj.shape
    assert L == len(scale_ys)
    sin = traj[..., -1:].sin()
    cos = traj[..., -1:].cos()
    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    x_diff = traj[..., 0:1].diff(n=1, dim=1, prepend=zero_pad) / scale_x_diff

    y = traj[..., 1:2]
    scale_ys_tensor = torch.tensor(scale_ys, dtype=traj.dtype, device=traj.device)
    scale_ys_tensor = scale_ys_tensor[None, :, None].repeat(B, 1, 1)
    y_norm = y / scale_ys_tensor

    return torch.cat([x_diff, y_norm, sin, cos], -1)


def cumsum_traj(norm_trajs):
    B, L, _ = norm_trajs.shape
    sin_values = norm_trajs[..., 2:3]
    cos_values = norm_trajs[..., 3:4]
    heading = torch.atan2(sin_values, cos_values)
    x = norm_trajs[..., 0:1].cumsum(1) * scale_x_diff

    scale_ys_tensor = torch.tensor(scale_ys, dtype=norm_trajs.dtype, device=norm_trajs.device)
    scale_ys_tensor = scale_ys_tensor[None, :, None].repeat(B, 1, 1)
    y = norm_trajs[..., 1:2] * scale_ys_tensor
    return torch.cat([x, y, heading], -1)


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dp_nlayers, input_dim, obs_len):
        super().__init__()
        self.dp_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), dp_nlayers
        )
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        token_len = obs_len + 1
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SimpleDiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self,
                sample,
                timestep,
                cond):
        B, HORIZON, DIM = sample.shape
        sample = sample.view(B, -1).float()
        input_emb = self.input_emb(sample)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([time_emb, cond], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
                              :, :tc, :
                              ]  # each position maps to a (learnable) vector
        x = cond_embeddings + position_embeddings
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb.unsqueeze(1)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        # (B,T,n_emb)
        x = self.dp_transformer(
            tgt=x,
            memory=memory,
        )
        # (B,T,n_emb)
        x = self.ln_f(x)
        x = self.output_emb(x)
        return x.squeeze(1).view(B, HORIZON, DIM)


class DPModel_traj_v1(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self._config = config
        assert config.backbone_type in ['vit', 'intern', 'vov', 'resnet', 'eva', 'moe', 'moe_ult32', 'swin']
        if config.backbone_type == 'eva':
            raise ValueError(f'{config.backbone_type} not supported')
        elif config.backbone_type == 'intern' or config.backbone_type == 'vov' or \
                config.backbone_type == 'swin' or config.backbone_type == 'vit' or config.backbone_type == 'resnet':
            self._backbone = HydraBackbone(config)

        img_num = 2 if config.use_back_view else 1
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num + 1, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        self._trajectory_head = DPHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        self.temporal_fuse = TemporalAttention(embed_dims=config.tf_d_model)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(self._config.seq_len * config.tf_d_model, config.tf_d_model * 4, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(config.tf_d_model * 4, config.tf_d_model, kernel_size=3, stride=1, padding=1),
        )

    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features).flatten(-2, -1)
        img_features = img_features.permute(0, 2, 1)
        return img_features

    def get_feats_1frame(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features)
        return img_features

    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"][0]

        batch_size = status_feature.shape[0]
        assert (camera_feature[-1].shape[0] == batch_size)

        video_feats = []
        assert self._config.seq_len == 2
        for frame_idx in range(self._config.seq_len):
            feat = self.get_feats_1frame(camera_feature[frame_idx])
            if frame_idx == 0:
                feat = feat.detach()
            video_feats.append(feat)
        # temporal fusion
        video_feats = torch.stack(video_feats, dim=1)
        video_feats = self.temporal_fuse(video_feats)
        video_feats = self.temporal_fusion(video_feats).flatten(-2, -1).permute(0, 2, 1)

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        keyval = video_feats
        keyval = torch.concatenate([keyval, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval)

        output.update(trajectory)
        output['env_kv'] = keyval

        return output


class DPHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DPConfig = None
                 ):
        super().__init__()
        self.config = config
        self.noise_scheduler = DDPMScheduler(
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
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps

    def forward(self, kv) -> Dict[str, torch.Tensor]:
        B = kv.shape[0]
        result = {}
        if not self.training:
            NUM_PROPOSALS = self.config.num_proposals

            condition = kv.repeat_interleave(NUM_PROPOSALS, dim=0)

            noise = torch.randn(
                size=(B * NUM_PROPOSALS, HORIZON, ACTION_DIM),
                dtype=condition.dtype,
                device=condition.device,
            )

            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            for t in self.noise_scheduler.timesteps:
                model_output = self.transformer_dp(
                    noise,
                    t,
                    condition
                )
                noise = self.noise_scheduler.step(
                    model_output, t, noise
                ).prev_sample
            traj = cumsum_traj(noise)
            result['dp_pred'] = traj.view(B, NUM_PROPOSALS, HORIZON, ACTION_DIM_ORI)

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
