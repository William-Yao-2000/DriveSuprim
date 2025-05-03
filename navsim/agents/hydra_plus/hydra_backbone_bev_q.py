"""
Implements the TransFuser vision backbone.
"""

import timm
import torch
from torch import nn

from navsim.agents.backbones.internimage import InternImage
from navsim.agents.backbones.swin import SwinTransformerBEVFT
from navsim.agents.backbones.vov import VoVNet
from navsim.agents.hydra_plus.hydra_config import HydraConfig
from navsim.agents.utils.vit import DAViT


class HydraBackboneBEVQ(nn.Module):
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
        else:
            raise ValueError

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

        self.bev_h, self.bev_w = 8, 8
        self.bev_queries = nn.Embedding(
            self.bev_h * self.bev_w, self.img_feat_c
        )
        self.pos_emb = nn.Embedding(
            (config.img_vert_anchors * config.img_horz_anchors +
             config.img_vert_anchors * int(config.img_horz_anchors * 1920 / 4096)) * 2, self.img_feat_c
        )
        self.bev_emb = nn.Embedding(
            self.bev_h * self.bev_w, self.img_feat_c
        )

        self.fusion_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.img_feat_c,
                nhead=16,
                dim_feedforward=self.img_feat_c * 4,
                dropout=0.0,
                batch_first=True
            ), self.config.fusion_layers
        )

        channel = self.config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        # top down
        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height
                    // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width
                    // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

            # lateral
            self.c5_conv = nn.Conv2d(
                self.img_feat_c,
                channel,
                (1, 1),
            )

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

    def encode_img(self, img):
        image_features = self.image_encoder(img)[-1]
        return image_features.flatten(-2, -1).permute(0, 2, 1)

    def forward(self, image_front, image_back, image_left, image_right):
        B = image_front.shape[0]

        image_features_front = self.encode_img(image_front)
        image_features_back = self.encode_img(image_back)
        image_features_left = self.encode_img(image_left)
        image_features_right = self.encode_img(image_right)

        img_tokens = torch.cat([image_features_front, image_features_back, image_features_left, image_features_right], 1)
        bev_tokens = self.bev_queries.weight[None].repeat(B, 1, 1)

        bev_tokens = self.fusion_decoder(
            tgt=bev_tokens + self.bev_emb.weight[None].repeat(B, 1, 1),
            memory=img_tokens + self.pos_emb.weight[None].repeat(B, 1, 1)
        )

        up_bev_tokens = self.top_down(
            bev_tokens.permute(0, 2, 1).view(B, self.img_feat_c, self.bev_h, self.bev_w)
        )
        return img_tokens, bev_tokens, up_bev_tokens
