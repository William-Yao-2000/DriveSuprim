"""
Implements the TransFuser vision backbone.
"""
import os

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

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
        elif config.backbone_type == 'resnet34':
            self.image_encoder = timm.create_model(
                'resnet34', pretrained=False, features_only=True
            )
            vit_channels = 512
        elif config.backbone_type == 'resnet50':
            self.image_encoder = timm.create_model(
                'resnet50', pretrained=False, features_only=True
            )
            vit_channels = 2048
        else:
            raise ValueError

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

    def forward(self, image):
        image_features = self.image_encoder(image)[-1]
        return self.avgpool_img(image_features)
    
    def forward_tup(self, image, **kwargs):
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()
        
        if isinstance(self.image_encoder, DAViT):
            image_feat_tup = self.image_encoder(image, **kwargs)[-1]
        else:
            image_feat = self.image_encoder(image)[-1]
            class_feat = image_feat.mean(dim=(-1, -2))
            image_feat_tup = (image_feat, class_feat)
        
        if self.config.lab.use_higher_res_feat_in_refinement:
            return (self.avgpool_img(image_feat_tup[0]), image_feat_tup[1], image_feat_tup[0])
        else:
            return (self.avgpool_img(image_feat_tup[0]), image_feat_tup[1])
