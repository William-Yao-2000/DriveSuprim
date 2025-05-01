"""
Implements the TransFuser vision backbone.
"""
import os

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from navsim.agents.backbones.eva import EVAViT
from navsim.agents.backbones.internimage import InternImage
from navsim.agents.backbones.swin import SwinTransformerBEVFT
from navsim.agents.backbones.vov import VoVNet
from navsim.agents.hydra_ssl.hydra_config_ssl import HydraConfigSSL
from navsim.agents.transfuser.transfuser_backbone import GPT
from navsim.agents.utils.vit import DAViT


class HydraBackbonePE(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config: HydraConfigSSL):

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
                depths=[2,2,18,2],
                drop_path_rate=0.35,
                embed_dims=192,
                init_cfg=dict(
                    checkpoint=config.swin_ckpt,
                    type='Pretrained'
                ),
                num_heads=[6,12,24,48],
                out_indices=[3],
                patch_norm=True,
                window_size=[16,16,16,16],
                use_abs_pos_embed=True,
                return_stereo_feat=False,
                output_missing_index_as_none=False
            )
            vit_channels = 1536
        elif config.backbone_type == 'vit':
            self.image_encoder = DAViT(ckpt=config.vit_ckpt)
            vit_channels = 1024
        elif config.backbone_type == 'sptr':
            """
            usage in config:
            
            camera_width: 2048
            camera_height: 512
            img_vert_anchors: 16
            img_horz_anchors: 64
            backbone_type: 'sptr'
            lr_mult_backbone: 0.1
            sptr_ckpt: ${oc.env:OPENSCENE_DATA_ROOT}/models/sptr_vit.pth
            
            # link for sptr_vit.pth: 
            wget https://github.com/exiawsh/storage/releases/download/v1.0/repdetr3d_eva02_800_bs2_seq_24e.pth
            mv repdetr3d_eva02_800_bs2_seq_24e.pth sptr_vit.pth
            """

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
        elif config.backbone_type == 'resnet':
            self.image_encoder = timm.create_model(
                'resnet34', pretrained=False, features_only=True
            )
            vit_channels = 512
        else:
            raise ValueError

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

    def forward(self, image):
        B, C, H, W = image.shape
        if self.backbone_type == 'vov' or self.backbone_type == 'vit':
            image_features = self.image_encoder(image)[-1]
        elif self.backbone_type == 'sptr':
            # split img into two patches
            half_w = W // 2
            image_left = image[..., :half_w]
            image_right = image[..., half_w:]
            image_features_left = self.image_encoder(image_left)[-1]
            image_features_right = self.image_encoder(image_right)[-1]
            image_features = torch.cat([
                image_features_left, image_features_right
            ], -1)
        else:
            raise ValueError('Forward wrong backbone')
        return self.avgpool_img(image_features)

    def forward_tup(self, image, **kwargs):
        image_feat_tup = self.image_encoder(image, **kwargs)[-1]
        if self.config.lab.use_higher_res_feat_in_refinement:
            return (self.avgpool_img(image_feat_tup[0]), image_feat_tup[1], image_feat_tup[0])
        else:
            return (self.avgpool_img(image_feat_tup[0]), image_feat_tup[1])
