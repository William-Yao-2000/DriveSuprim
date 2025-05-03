"""
Implements the TransFuser vision backbone.
"""

import timm
import torch
from torch import nn

from navsim.agents.backbones.eva import EVAViT
from navsim.agents.backbones.internimage import InternImage
from navsim.agents.backbones.swin import SwinTransformerBEVFT
from navsim.agents.backbones.vov import VoVNet
from navsim.agents.hydra_plus.hydra_config import HydraConfig
from navsim.agents.utils.vit import DAViT


class HydraBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config: HydraConfig):

        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type
        if config.backbone_type == 'intern':
            self.image_encoder = InternImage(init_cfg=dict(type='Pretrained',
                                                           checkpoint=config.intern_ckpt
                                                           ),
                                             frozen_stages=2)
            # scale_4_c = 2560
            vit_channels = 2560
            self.image_encoder.init_weights()
        elif config.backbone_type == 'vov':
            self.image_encoder = VoVNet(
                spec_name='V-99-eSE',
                out_features=['stage4', 'stage5'],
                norm_eval=True,
                with_cp=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=config.vov_ckpt,
                    prefix='img_backbone.'
                )
            )
            # scale_4_c = 1024
            vit_channels = 1024
            self.image_encoder.init_weights()
        elif config.backbone_type == 'swin':
            self.image_encoder = SwinTransformerBEVFT(
                with_cp=True,
                convert_weights=False,
                depths=[2, 2, 18, 2],
                drop_path_rate=0.35,
                embed_dims=192,
                init_cfg=dict(
                    checkpoint=config.swin_ckpt,
                    type='Pretrained'
                ),
                num_heads=[6, 12, 24, 48],
                out_indices=[3],
                patch_norm=True,
                window_size=[16, 16, 16, 16],
                use_abs_pos_embed=True,
                return_stereo_feat=False,
                output_missing_index_as_none=False
            )
            vit_channels = 1536
        elif config.backbone_type == 'vit':
            self.image_encoder = DAViT(ckpt=config.vit_ckpt)
            vit_channels = 1024
        elif config.backbone_type == 'resnet':
            self.image_encoder = timm.create_model(
                'resnet34', pretrained=False, features_only=True
            )
            vit_channels = 512
        elif config.backbone_type == 'sptr':
            img_vit_size = (config.camera_height, config.camera_width)
            self.image_encoder = EVAViT(
                img_size=img_vit_size[0],  # img_size for short side
                patch_size=16,
                window_size=16,
                global_window_size=img_vit_size[0] // 16,
                # If use square image (e.g., set global_window_size=0, else global_window_size=img_size // 16)
                in_chans=3,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4 * 2 / 3,
                window_block_indexes=(
                        list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(
                    range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
                ),
                qkv_bias=True,
                drop_path_rate=0.3,
                with_cp=True,
                flash_attn=False,
                xformers_attn=True
            )
            self.image_encoder.init_weights(config.sptr_ckpt)
            vit_channels = 1024

        else:
            raise ValueError('Unsupported backbone in hydra_backbone')

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

    def forward(self, image):
        B, C, H, W = image.shape
        if self.backbone_type == 'vov':
            image_features = self.image_encoder(image)[-1]
        elif self.backbone_type == 'sptr':
            half_w = W // 2
            image_left = image[..., :half_w]
            image_right = image[..., half_w:]
            image_features_left = self.image_encoder(image_left)[-1]
            image_features_right = self.image_encoder(image_right)[-1]
            image_features = torch.cat([
                image_features_left, image_features_right
            ], -1)
        elif self.backbone_type == 'vit':
            quarter_w = W // 4
            image_1 = image[..., :quarter_w]
            image_2 = image[..., quarter_w:2 * quarter_w]
            image_3 = image[..., 2 * quarter_w:3 * quarter_w]
            image_4 = image[..., 3 * quarter_w:]
            image_f_1 = self.image_encoder(image_1)[-1]
            image_f_2 = self.image_encoder(image_2)[-1]
            image_f_3 = self.image_encoder(image_3)[-1]
            image_f_4 = self.image_encoder(image_4)[-1]
            image_features = torch.cat([
                image_f_1, image_f_2, image_f_3, image_f_4
            ], -1)
        else:
            raise ValueError('Forward wrong backbone')
        return self.avgpool_img(image_features)
