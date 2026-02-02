import copy
from itertools import combinations
import numpy as np
import os, pickle
from sklearn.cluster import KMeans
from typing import Dict, List

import torch
import torch.nn as nn

from navsim.agents.drivesuprim.drivesuprim_backbone_pe import DriveSuprimBackbonePE
from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.transfuser.transfuser_model import AgentHead
from navsim.agents.utils.attn import MemoryEffTransformer
from navsim.agents.utils.nerf import nerf_positional_encoding


class DriveSuprimModel(nn.Module):
    def __init__(self, config: DriveSuprimConfig):
        super().__init__()

        self._config = config
        assert config.backbone_type in ['vit', 'intern', 'vov', 'resnet34', 'resnet50', 'eva', 'moe', 'moe_ult32', 'swin', 'sptr']
        if config.backbone_type == 'eva':
            raise ValueError(f'{config.backbone_type} not supported')
        elif config.backbone_type == 'intern' or config.backbone_type == 'vov' or config.backbone_type == 'swin' or config.backbone_type == 'vit' or \
             config.backbone_type in ('resnet34', 'resnet50') or config.backbone_type == 'sptr':
            self._backbone = DriveSuprimBackbonePE(config)

        img_num = 1
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num, config.tf_d_model
        )

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )

        self.use_multi_stage = self._config.refinement.use_multi_stage
        if self.use_multi_stage:
            if self._config.refinement.refinement_approach == 'transformer_decoder':
                self._trajectory_offset_head = RefineTrajHead(
                    d_ffn=config.tf_d_ffn,
                    d_model=config.tf_d_model,
                    nhead=config.vadv2_head_nhead,
                    d_backbone=self._backbone.img_feat_c,
                    num_stage=config.refinement.num_refinement_stage,
                    stage_layers=config.refinement.stage_layers,
                    topks=config.refinement.topks,
                    config=config
                )
            else:
                raise NotImplementedError

    def img_feat_blc_dict(self, camera_feature, **kwargs):
        img_features = self._backbone(camera_feature, **kwargs)
        img_feat_dict = {
            'patch_token': img_features,  # [bs, c_img, h_avg, w_avg]
        }
        img_features = self.downscale_layer(img_features).flatten(-2, -1)  # [bs, c, h_avg * w_avg]
        img_features = img_features.permute(0, 2, 1)  # [bs, h_avg * w_avg, c]
        img_feat_dict['avg_feat'] = img_features
        return img_feat_dict

    def forward_features_list(self, x_list):
        return [self.forward(x) for x in x_list]

    def forward(self,
                features: Dict[str, torch.Tensor],
                masks=None,
                tokens=None) -> Dict[str, torch.Tensor]:
        
        output: Dict[str, torch.Tensor] = {}

        camera_feature: List[torch.Tensor] = features["camera_feature"]  # List[torch.Tensor], len == seq_len, tensor.shape == [bs, 3, h, w]
        status_feature: torch.Tensor = features["status_feature"][0]  # tensor.shape == [bs, 8] (features["status_feature"][0] picks present status)
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]  # [bs, 3, h, w], [-1] means present frame camera input

        img_feat_dict = self.img_feat_blc_dict(camera_feature, masks=masks, return_class_token=False)

        status_encoding = self._status_encoding(status_feature)  # [bs, 8] -> [bs, c]

        keyval = img_feat_dict.pop('avg_feat')  # [bs, h_avg * w_avg, c]
        keyval += self._keyval_embedding.weight[None, ...]  # [bs, h_avg * w_avg, c]

        output.update(img_feat_dict)
        trajectory = self._trajectory_head(keyval, status_encoding, tokens=tokens)

        if self.use_multi_stage:
            img_feat = img_feat_dict['patch_token']  # [bs, c_vit, w, h]
            final_traj = self._trajectory_offset_head(img_feat, trajectory['refinement'])
            trajectory['final_traj'] = final_traj

        output.update(trajectory)

        return output


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DriveSuprimConfig = None
                 ):
        super().__init__()
        self._config = config
        self._num_poses = num_poses
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )

        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'history_comfort': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'imi': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
        })

        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )

        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, img_feature, status_encoding, tokens=None) -> Dict[str, torch.Tensor]:

        vocab = self.vocab.data  # [n_vocab, 40, 3]
        L, HORIZON, _ = vocab.shape
        B = img_feature.shape[0]

        if self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]  # [1, n_vocab, c]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)  # [bs, n_vocab, c]
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)

        tr_out = self.transformer(embedded_vocab, img_feature)  # [bs, n_vocab, c]
        dist_status = tr_out + status_encoding.unsqueeze(1)  # [bs, n_vocab, c]
        result = {}
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        scores = (
            0.02 * result['imi'].softmax(-1).log() +
            0.1 * result['traffic_light_compliance'].sigmoid().log() +
            0.5 * result['no_at_fault_collisions'].sigmoid().log() +
            0.5 * result['drivable_area_compliance'].sigmoid().log() +
            0.3 * result['driving_direction_compliance'].sigmoid().log() +
            6.0 * (5.0 * result['time_to_collision_within_bound'].sigmoid() +
                   5.0 * result['ego_progress'].sigmoid() +
                   2.0 * result['lane_keeping'].sigmoid() +
                   1.0 * result['history_comfort'].sigmoid()
                  ).log()
        )  # [bs, n_vocab]
        
        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices

        if self._config.refinement.use_multi_stage:
            topk_str = str(self._config.refinement.topks)
            topk = int(topk_str.split('+')[0])
            topk_values, topk_indices = torch.topk(scores, k=topk, dim=1)
            result['refinement'] = []  # dicts of different refinement stages
            _dict = {}
            _dict["trajs"] = self.vocab.data[topk_indices]
            # Gather the statuses for the top-k trajectories
            batch_indices = torch.arange(B, device=topk_indices.device).view(-1, 1).expand(-1, topk)
            _dict["trajs_status"] = dist_status[batch_indices, topk_indices]
            _dict['indices_absolute'] = topk_indices

            # Store the scores for each top-k trajectory
            _dict['coarse_score'] = {}
            for score_key in self.heads.keys():
                _dict['coarse_score'][score_key] = result[score_key][batch_indices, topk_indices]

            result['refinement'].append(_dict)
        
        return result


class RefineTrajHead(nn.Module):
    def __init__(self, d_ffn: int, d_model: int, nhead: int, d_backbone: int,
                 num_stage: int, stage_layers: str, topks: str,
                 config: DriveSuprimConfig = None
                 ):
        super().__init__()
        
        stage_layers = str(stage_layers)
        topks = str(topks)
        
        self._config = config
        self.num_stage = num_stage  # the number of **refinement** stages, we choose to use only 1 refinement stage (8192->256)
        self.stage_layers = [int(sl) for sl in stage_layers.split('+')]
        self.topks = [int(topk) for topk in topks.split('+')]
        assert len(self.stage_layers) == num_stage and len(self.topks) == num_stage
        # self.nlayers = sum(self.stage_layers)

        self.use_mid_output = config.refinement.use_mid_output
        self.use_separate_stage_heads = config.refinement.use_separate_stage_heads

        downscale_layer = nn.Conv2d(d_backbone, d_model, kernel_size=1)
        if self.use_separate_stage_heads:
            self.downscale_layers = nn.ModuleList([copy.deepcopy(downscale_layer) for _ in range(num_stage)])
        else:
            self.downscale_layers = nn.ModuleList([downscale_layer for _ in range(num_stage)])

        transformer_blocks = [TransformerDecoder_v2(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), layer
        ) for layer in self.stage_layers]
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'history_comfort': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
        })

        if self._config.refinement.use_imi_learning_in_refinement:
            heads['imi'] = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )

        if self.use_separate_stage_heads:
            self.multi_stage_heads = nn.ModuleList([copy.deepcopy(heads) for _ in range(num_stage)])
        else:
            self.multi_stage_heads = nn.ModuleList([heads for _ in range(num_stage)])

        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )

    def forward(self, img_feat, refinement_dict) -> Dict[str, torch.Tensor]:

        B = img_feat.shape[0]
        
        for i in range(self.num_stage):
            _img_feat_fg = self.downscale_layers[i](img_feat).flatten(2)
            _img_feat_fg = _img_feat_fg.permute(0, 2, 1)  # [bs, h_avg * w_avg, c]
            status_encoding = refinement_dict[-1]['trajs_status']  # [bs, topk_stage_i, c]
            tr_out_lst = self.transformer_blocks[i](status_encoding, _img_feat_fg)  # [layer_stage_i, bs, topk_stage_i, c]

            # Compute scores for each refinement decoder layer
            layer_results = []
            for j, dist_status in enumerate(tr_out_lst):
                layer_result = {}
                for k, head in self.multi_stage_heads[i].items():
                    layer_result[k] = head(dist_status).squeeze(-1)
                layer_results.append(layer_result)
            
            if not self.use_mid_output:
                layer_results = layer_results[-1:]
            refinement_dict[-1]['layer_results'] = layer_results
            
            last_layer_result = layer_results[-1]

            scores = (
                0.1 * last_layer_result['traffic_light_compliance'].sigmoid().log() +
                0.5 * last_layer_result['no_at_fault_collisions'].sigmoid().log() +
                0.5 * last_layer_result['drivable_area_compliance'].sigmoid().log() +
                0.3 * last_layer_result['driving_direction_compliance'].sigmoid().log() +
                6.0 * (5.0 * last_layer_result['time_to_collision_within_bound'].sigmoid() +
                       5.0 * last_layer_result['ego_progress'].sigmoid() +
                       2.0 * last_layer_result['lane_keeping'].sigmoid() +
                       1.0 * last_layer_result['history_comfort'].sigmoid()
                      ).log()
            )

            if self._config.refinement.use_imi_learning_in_refinement:
                scores += 0.02 * last_layer_result['imi'].softmax(-1).log()

            if i != self.num_stage-1:
                _next_topk = self.topks[i+1]
                _, select_indices = torch.topk(scores, k=_next_topk, dim=1)
                batch_indices = torch.arange(B, device=select_indices.device).view(-1, 1).expand(-1, _next_topk)

                _next_layer_dict = {}
                _next_layer_dict["trajs"] = refinement_dict[-1]['trajs'][batch_indices, select_indices]
                _next_layer_dict["trajs_status"] = tr_out_lst[-1][batch_indices, select_indices]
                _next_layer_dict['indices_absolute'] = refinement_dict[-1]['indices_absolute'][batch_indices, select_indices]
                refinement_dict.append(_next_layer_dict)
            
            else:
                select_indices = scores.argmax(1)
                batch_indices = torch.arange(B, device=select_indices.device)
                final_traj = refinement_dict[-1]['trajs'][batch_indices, select_indices]  # [bs, 40, 3]
                # filtered_scores = scores
        
        return final_traj


class TransformerDecoder_v2(nn.TransformerDecoder):

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None):
        output = tgt

        output_lst = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            output_lst.append(output)

        return output_lst
