from typing import Dict
import os, pickle

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.hydra.hydra_backbone_pe import HydraBackbonePE
from navsim.agents.hydra_ssl.hydra_config_ssl import HydraConfigSSL
from navsim.agents.transfuser.transfuser_model import AgentHead
from navsim.agents.utils.attn import MemoryEffTransformer
from navsim.agents.utils.nerf import nerf_positional_encoding
from navsim.agents.vadv2.vadv2_config import Vadv2Config


class HydraModel(nn.Module):
    def __init__(self, config: HydraConfigSSL):
        super().__init__()

        self._query_splits = [
            config.num_bounding_boxes,
        ]

        self._config = config
        assert config.backbone_type in ['vit', 'intern', 'vov', 'resnet', 'eva', 'moe', 'moe_ult32', 'swin']
        if config.backbone_type == 'eva':
            raise ValueError(f'{config.backbone_type} not supported')
        elif config.backbone_type == 'intern' or config.backbone_type == 'vov' or \
            config.backbone_type == 'swin' or config.backbone_type == 'vit' or config.backbone_type == 'resnet':
            self._backbone = HydraBackbonePE(config)

        img_num = 2 if config.use_back_view else 1
        # import pdb; pdb.set_trace()
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        # self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)


        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        # self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        # self._agent_head = AgentHead(
        #     num_agents=config.num_bounding_boxes,
        #     d_ffn=config.tf_d_ffn,
        #     d_model=config.tf_d_model,
        # )

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )

        self.use_2_stage = self._config.refinement.use_2_stage
        if self.use_2_stage:
            if self._config.refinement.refinement_approach == 'offset_decoder':
                self._trajectory_offset_head = TrajOffsetHead(
                    d_ffn=config.tf_d_ffn,
                    d_model=config.tf_d_model,
                    nhead=config.vadv2_head_nhead,
                    nlayers=config.refinement.n_offset_dec_layers,
                    d_backbone=self._backbone.img_feat_c,
                    config=config
                )
            else:
                raise NotImplementedError

        if self._config.lab.check_top_k_traj:
            self.test_full_vocab_pdm_score = pickle.load(
                open(self._config.lab.test_full_vocab_pdm_score_path, 'rb'))

    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)  # [b, c_img, h//32, w//32]
        img_features = self.downscale_layer(img_features).flatten(-2, -1)  # [b, c, h//32 * w//32]
        img_features = img_features.permute(0, 2, 1)  # [b, h//32 * w//32, c]
        return img_features

    def img_feat_blc_dict(self, camera_feature, **kwargs):
        img_feat_tup = self._backbone.forward_tup(camera_feature, **kwargs)
        img_feat_dict = {
            'patch_token': img_feat_tup[0],  # [b, c_img, h//32, w//32], the patch size of vit is only 16, but use a avg pooling
            'class_token': img_feat_tup[1],
        }
        img_features = img_feat_dict['patch_token']
        img_features = self.downscale_layer(img_features).flatten(-2, -1)  # [b, c, h//32 * w//32]
        img_features = img_features.permute(0, 2, 1)  # [b, h//32 * w//32, c]
        img_feat_dict['avg_feat'] = img_features
        return img_feat_dict

    def forward_features_list(self, x_list, mask_list):
        return [self.forward(x, m) for x, m in zip(x_list, mask_list)]

    def forward(self,
                features: Dict[str, torch.Tensor],
                masks=None,
                interpolated_traj=None,
                tokens=None) -> Dict[str, torch.Tensor]:
        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
            import pdb; pdb.set_trace()
        output: Dict[str, torch.Tensor] = {}

        camera_feature: torch.Tensor = features["camera_feature"]  # List[torch.Tensor], len == seq_len, tensor.shape == [b, 3, h, w]
        status_feature: torch.Tensor = features["status_feature"][0]  # List[torch.Tensor], len == seq_len, tensor.shape == [b, 8] (the [0] picks present status)
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]  # [b, 3, h, w]
        # todo temp fix!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # status_feature[:, 0] = 0.0
        # status_feature[:, 1] = 1.0
        # status_feature[:, 2] = 0.0
        # status_feature[:, 3] = 0.0
        
        batch_size = status_feature.shape[0]
        
        img_feat_dict = self.img_feat_blc_dict(camera_feature, masks=masks, return_class_token=True)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)  # [b, 8] -> [b, c]

        keyval = img_feat_dict.pop('avg_feat')  # [b, h//32 * w//32, c]
        keyval += self._keyval_embedding.weight[None, ...]  # [b, h//32 * w//32, c]

        # query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        # agents_query = self._tf_decoder(query, keyval)

        output.update(img_feat_dict)
        trajectory = self._trajectory_head(keyval, status_encoding, interpolated_traj)

        if self.use_2_stage:
            # result["topk_trajs"] = self.vocab.data[top_indices]
            # # Gather the statuses for the top-k trajectories
            # batch_indices = torch.arange(B, device=top_indices.device).view(-1, 1).expand(-1, topk)
            topk_trajs_status = trajectory['topk'].pop("topk_trajs_status")  # [bs, topk, c]
            bev_feat_fg = img_feat_dict['patch_token']  # [bs, c_vit, w, h]
            layer_results, selected_indices = self._trajectory_offset_head(bev_feat_fg, topk_trajs_status, trajectory['topk']['coarse_score'])
            trajectory['topk']['refinement_scores'] = layer_results
            # Get the best trajectory from the top-k based on refinement scores
            batch_indices = torch.arange(batch_size, device=selected_indices.device)
            trajectory['topk']['trajectory_2_stage'] = trajectory['topk']['topk_trajs'][batch_indices, selected_indices]


        if self._config.lab.check_top_k_traj:
            topk_selected_indices_bs = trajectory['topk_selected_indices']
            for i, (token, topk_selected_indices) in enumerate(zip(tokens, topk_selected_indices_bs)):
                # Get the scores for the topk trajectories
                topk_indices_cpu = topk_selected_indices.cpu().numpy()
                scores = self.test_full_vocab_pdm_score[token]['total'][topk_indices_cpu]
                best_idx = np.argmax(scores)
                # Update the trajectory for this sample to the best one according to PDM score
                trajectory['trajectory'][i] = trajectory["topk_trajs"][i][best_idx]

        output.update(trajectory)
        # agents = self._agent_head(agents_query)
        # output.update(agents)

        return output


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: HydraConfigSSL = None
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
            'noc': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'da':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'ttc': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'comfort': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'progress': nn.Sequential(
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

        self.inference_imi_weight = config.inference_imi_weight
        self.inference_da_weight = config.inference_da_weight
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )
        self.use_nerf = config.use_nerf

        if self.use_nerf:
            self.pos_embed = nn.Sequential(
                nn.Linear(1040, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * 3, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )

    def forward(self, bev_feature, status_encoding, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        # todo sinusoidal embedding
        # vocab: 4096, 40, 3
        # bev_feature: B, 32, C
        # embedded_vocab: B, 4096, C
        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_feature.shape[0]
        if self.use_nerf:
            vocab = torch.cat(
                [
                    nerf_positional_encoding(vocab[..., :2]),
                    torch.cos(vocab[..., -1])[..., None],
                    torch.sin(vocab[..., -1])[..., None],
                ], dim=-1
            )

        if self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)  # [b, n_vocab, c]
        tr_out = self.transformer(embedded_vocab, bev_feature)  # [b, n_vocab, c]
        dist_status = tr_out + status_encoding.unsqueeze(1)  # [b, n_vocab, c]
        result = {}
        # selected_indices: B,
        for k, head in self.heads.items():
            if k == 'imi':
                result[k] = head(dist_status).squeeze(-1)  # [b, n_vocab]
            else:
                result[k] = head(dist_status).squeeze(-1).sigmoid()
        scores = (
                0.05 * result['imi'].softmax(-1).log() +
                0.5 * result['noc'].log() +
                0.5 * result['da'].log() +
                8.0 * (5 * result['ttc'] + 2 * result['comfort'] + 5 * result['progress']).log()
        )
        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()
        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices

        if self._config.refinement.use_2_stage:
            assert self._config.lab.check_top_k_traj is False
            topk = self._config.refinement.num_top_k
            top_values, top_indices = torch.topk(scores, k=topk, dim=1)
            result['topk'] = {}
            result['topk']["topk_trajs"] = self.vocab.data[top_indices]
            # Gather the statuses for the top-k trajectories
            batch_indices = torch.arange(B, device=top_indices.device).view(-1, 1).expand(-1, topk)
            result['topk']["topk_trajs_status"] = dist_status[batch_indices, top_indices]
            result['topk']["topk_selected_indices"] = top_indices
            
            # Store the scores for each top-k trajectory
            result['topk']['coarse_score'] = {}
            for score_key in ['noc', 'da', 'ttc', 'comfort', 'progress']:
                result['topk']['coarse_score'][score_key] = result[score_key][batch_indices, top_indices]

        # debug
        if self._config.lab.check_top_k_traj:
            topk = self._config.lab.num_top_k
            top_values, top_indices = torch.topk(scores, k=topk, dim=1)
            result["topk_trajs"] = self.vocab.data[top_indices]
            result["topk_selected_indices"] = top_indices
        
        return result
    

class TrajOffsetHead(nn.Module):
    def __init__(self, d_ffn: int, d_model: int, nhead: int, nlayers: int, d_backbone: int,
                 config: HydraConfigSSL = None
                 ):
        super().__init__()
        self._config = config

        self.downscale_layer = nn.Conv2d(d_backbone, d_model, kernel_size=1)

        self.transformer = TransformerDecoder_v2(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), nlayers
        )

        self.heads = nn.ModuleDict({
            'noc': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'da':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'ttc': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'comfort': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
        })

        self.inference_da_weight = config.inference_da_weight
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )

    def forward(self, bev_feat_fg, status_encoding, coarse_score, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        # bev_feature_fg (bev_feature_fine_grained): bs, c_vit, h, w
        # status_encoding: bs, topk, c
        # coarse_scores: dict

        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()

        bev_feat_fg = self.downscale_layer(bev_feat_fg).flatten(2)

        tr_out_lst = self.transformer(status_encoding, bev_feat_fg)  # [b, n_vocab, c]

        # Compute scores for each layer
        layer_results = []
        # Initialize reference scores from coarse_score
        reference = coarse_score
        for i, dist_status in enumerate(tr_out_lst):
            layer_result = {}
            for k, head in self.heads.items():
                # Compute offset and apply to reference using inverse sigmoid
                offset = head(dist_status).squeeze(-1)
                # inverse_sigmoid(reference) + offset, then sigmoid
                reference_inv = _inverse_sigmoid(reference[k])
                layer_result[k] = torch.sigmoid(reference_inv + offset)
                # Update reference for next layer
                reference[k] = layer_result[k]
            layer_results.append(layer_result)

        # Calculate final scores using the last layer's results
        last_layer = layer_results[-1]
        scores = (
                0.5 * last_layer['noc'].log() +
                0.5 * last_layer['da'].log() +
                8.0 * (5 * last_layer['ttc'] + 2 * last_layer['comfort'] + 5 * last_layer['progress']).log()
        )

        selected_indices = scores.argmax(1)
        return layer_results, selected_indices
    

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

def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)