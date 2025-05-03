import os
import pickle
from pathlib import Path
from typing import Any, Union
from typing import Dict
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.dp.dp_model import DPModel
from navsim.agents.dp.dp_model_traj_v1 import DPModel_traj_v1
from navsim.agents.dp.dp_model_traj_v1_bev import DPModel_traj_v1_bev
from navsim.agents.dp.dp_model_traj_v1_bev_ddpmft import DPModel_traj_v1_bev_ddpmft, DPHeadFt
from navsim.agents.dp.dp_model_traj_v2_bev import DPModel_traj_v2_bev
from navsim.agents.dp.dp_model_traj_v2_bev_ddpmft import DPModel_traj_v2_bev_ddpmft
from navsim.agents.dp.dp_model_traj_v2_bev_q import DPModel_traj_v2_bev_q
from navsim.agents.dp.reward_utils import hydra_eval_dp, pdm_eval_dp_multithread
from navsim.agents.hydra_plus.hydra_config import HydraConfig
from navsim.agents.hydra_plus.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.agents.hydra_plus.hydra_model import HydraModel
from navsim.agents.hydra_plus.hydra_plus_agent import hydra_kd_imi_agent_loss
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

GAMMA = 0.99


class ReferenceModelUpdateCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print('Updating reference model')
        pl_module.agent.model._trajectory_head.update_reference_model()


def dp_loss_bev(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor],
        config: DPConfig, traj_head
):
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    dp_loss = traj_head.get_dp_loss(predictions['env_kv'], target_traj.float())
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())

    dp_loss = dp_loss * 10.0
    bev_semantic_loss = bev_semantic_loss * 10.0
    loss = (
            dp_loss +
            bev_semantic_loss
    )
    return loss, {
        'dp_loss': dp_loss,
        'bev_semantic_loss': bev_semantic_loss
    }


def dp_loss_rl(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor],
        config: DPConfig,
        traj_head: DPHeadFt,
        hydra_model,
        features,
        metric_cache_loader=None,
        simulator=None,
        scorer=None,
        traffic_agents=None,
        tokens=None,
        reinforce=False
):
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    dp_loss = traj_head.get_dp_loss(predictions['env_kv'], target_traj.float())
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    dp_loss = dp_loss
    bev_semantic_loss = bev_semantic_loss

    # (B, N, 8, 3)
    trajectories = predictions['dp_pred']
    # (B*N, num_steps + 1, 8, 4)
    latents_full = torch.stack(
        predictions['latents'], dim=1
    )
    latents = latents_full[:, :-1]
    next_latents = latents_full[:, 1:]
    # (B*N, num_steps)
    log_probs_all = torch.stack(predictions['log_probs'], dim=1)
    timesteps = predictions['timesteps'].to(log_probs_all.device)
    log_probs_all = log_probs_all.clamp(min=-5, max=2)

    if config.reward_source == 'hydra':
        rewards = hydra_eval_dp(
            features,
            trajectories,
            hydra_model
        )['overall_scores'].nan_to_num(nan=-10.0, posinf=10.0, neginf=-10.0)
    else:
        rewards = pdm_eval_dp_multithread(
            tokens,
            trajectories,
            metric_cache_loader=metric_cache_loader,
            scorer=scorer,
            simulator=simulator,
            traffic_agents_policy_stage_one=traffic_agents
        )['pdm_score']

    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-7)
    advantages = torch.clamp(
        advantages,
        -config.adv_clip_max,
        config.adv_clip_max,
    ).flatten()

    ppo_loss = 0.0
    noise_scheduler = traj_head.noise_scheduler
    num_train_steps = noise_scheduler.config.num_train_timesteps
    for t in range(num_train_steps):
        model_output = traj_head.transformer_dp(
            latents[:, t],
            timesteps[t],
            predictions['condition']
        )
        _, log_prob_curr = noise_scheduler.step(
            model_output, timesteps[t].item(), latents[:, t], prev_sample=next_latents[:, t],
            stddev_clip=config.stdev_clip
        )
        log_prob_curr = log_prob_curr.clamp(min=-5, max=2)
        advantages_discounted = advantages * (GAMMA ** (num_train_steps - 1 - t))

        if reinforce:
            reinforce_loss = -advantages_discounted * log_prob_curr
            loss_curr = torch.mean(reinforce_loss)
        else:
            ratio = torch.exp(log_prob_curr - log_probs_all[:, t])
            unclipped_loss = -advantages_discounted * ratio
            clipped_loss = -advantages_discounted * torch.clamp(
                ratio,
                1.0 - config.clip_range,
                1.0 + config.clip_range,
            )
            loss_curr = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        ppo_loss += loss_curr

        # debug
        # print(f'ppo: {ppo_loss_curr}')
        # print(f'kl: {0.5 * torch.mean((log_prob_curr - log_probs_all[:, t]) ** 2).item()}')
        # print(f'clipfrac: {torch.mean((torch.abs(ratio - 1.0) > config.clip_range).float()).item()}')

    bev_semantic_loss *= 10.0
    ppo_loss *= 1.0
    dp_loss *= 10.0

    # todo BC Loss
    loss = (
            ppo_loss +
            bev_semantic_loss +
            dp_loss
    )
    return loss, {
        'dp_loss': dp_loss,
        'ppo_loss': ppo_loss,
        'rewards_mean': rewards.mean(),
        'bev_semantic_loss': bev_semantic_loss
    }


def dp_loss_rl_filter(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor],
        config: DPConfig,
        traj_head: DPHeadFt,
        hydra_model,
        features,
        metric_cache_loader=None,
        simulator=None,
        scorer=None,
        traffic_agents=None,
        tokens=None,
        reinforce=False,
        hydra_config=None,
        hydra_gt_scores=None
):
    B = len(tokens)
    noise_scheduler = traj_head.noise_scheduler
    num_train_steps = noise_scheduler.config.num_train_timesteps

    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    dp_loss = traj_head.get_dp_loss(predictions['env_kv'], target_traj.float())
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    dp_loss = dp_loss
    bev_semantic_loss = bev_semantic_loss

    # (B, N, 8, 3)
    trajectories = predictions['dp_pred']
    # (B*N, num_steps + 1, 8, 4)
    latents_full = torch.stack(
        predictions['latents'], dim=1
    )
    latents = latents_full[:, :-1]
    next_latents = latents_full[:, 1:]
    # (B*N, num_steps)
    log_probs_all = torch.stack(predictions['log_probs'], dim=1).clamp(min=-5, max=2)
    timesteps = predictions['timesteps'].to(log_probs_all.device)

    hydra_eval_results = hydra_eval_dp(
        features,
        trajectories,
        hydra_model,
        open_hydra=config.open_hydra,
        topk=config.hydra_topk
    )
    hydra_scores = hydra_eval_results['overall_log_scores']
    # B, 10
    hydra_topk_inds = hydra_scores.topk(k=config.hydra_topk, dim=1).indices

    # 扩展索引形状为 [B, 10, 1, 1]，然后扩展为 [B, 10, 8, 3]
    HORIZON = trajectories.shape[2]
    TRAJ_DIM = trajectories.shape[3]
    LATENT_DIM = latents.shape[3]
    TOKEN_LEN = predictions['condition'].shape[1]
    C = predictions['condition'].shape[2]

    # 使用gather收集数据
    trajectories_k = trajectories.gather(
        dim=1,
        index=hydra_topk_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, HORIZON, TRAJ_DIM)
    )
    latents_k = latents.view(B, config.num_proposals, num_train_steps, HORIZON, LATENT_DIM).gather(
        dim=1,
        index=hydra_topk_inds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_train_steps, HORIZON,
                                                                               LATENT_DIM)
    ).view(-1, num_train_steps, HORIZON, LATENT_DIM)
    latents_next_k = next_latents.view(B, config.num_proposals, num_train_steps, HORIZON, LATENT_DIM).gather(
        dim=1,
        index=hydra_topk_inds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_train_steps, HORIZON,
                                                                               LATENT_DIM)
    ).view(-1, num_train_steps, HORIZON, LATENT_DIM)
    log_probs_all_k = log_probs_all.view(B, config.num_proposals, num_train_steps).gather(
        dim=1,
        index=hydra_topk_inds.unsqueeze(-1).expand(-1, -1, num_train_steps)
    ).view(-1, num_train_steps)
    conditions_k = predictions['condition'].view(B, config.num_proposals, TOKEN_LEN, C).gather(
        dim=1,
        index=hydra_topk_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, TOKEN_LEN, C)
    ).view(-1, TOKEN_LEN, C)

    pdm_eval_results = pdm_eval_dp_multithread(
        tokens,
        trajectories_k,
        metric_cache_loader=metric_cache_loader,
        scorer=scorer,
        simulator=simulator,
        traffic_agents_policy_stage_one=traffic_agents
    )
    rewards = pdm_eval_results['pdm_score']

    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-7)
    advantages = torch.clamp(
        advantages,
        -config.adv_clip_max,
        config.adv_clip_max,
    ).flatten()

    ppo_loss = 0.0

    for t in range(num_train_steps):
        model_output = traj_head.transformer_dp(
            latents_k[:, t],
            timesteps[t],
            conditions_k
        )
        _, log_prob_curr = noise_scheduler.step(
            model_output, timesteps[t].item(), latents_k[:, t], prev_sample=latents_next_k[:, t],
            stddev_clip=config.stdev_clip
        )
        log_prob_curr = log_prob_curr.clamp(min=-5, max=2)
        advantages_discounted = advantages * (GAMMA ** (num_train_steps - 1 - t))

        if reinforce:
            reinforce_loss = -advantages_discounted * log_prob_curr
            loss_curr = torch.mean(reinforce_loss)
        else:
            # PPO with trust region
            ratio = torch.exp(log_prob_curr - log_probs_all_k[:, t])
            unclipped_loss = -advantages_discounted * ratio
            clipped_loss = -advantages_discounted * torch.clamp(
                ratio,
                1.0 - config.clip_range,
                1.0 + config.clip_range,
            )
            loss_curr = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        ppo_loss += loss_curr

        # debug
        # print(f'ppo: {ppo_loss_curr}')
        # print(f'kl: {0.5 * torch.mean((log_prob_curr - log_probs_all[:, t]) ** 2).item()}')
        # print(f'clipfrac: {torch.mean((torch.abs(ratio - 1.0) > config.clip_range).float()).item()}')

    bev_semantic_loss *= 10.0
    ppo_loss *= 1.0
    dp_loss *= 10.0

    loss = (
            ppo_loss +
            bev_semantic_loss +
            dp_loss
    )
    loss_dict = {
        'dp_loss': dp_loss,
        'ppo_loss': ppo_loss,
        'rewards_mean': rewards.mean(),
        'bev_semantic_loss': bev_semantic_loss
    }

    if config.open_hydra:
        distill_metrics = set(hydra_gt_scores.keys()) & set(hydra_eval_results.keys())
        # create gt for hydra: 16384 + 10 proposals
        for m in distill_metrics:
            hydra_gt_scores[m] = torch.cat(
                [hydra_gt_scores[m].float(), pdm_eval_results[m]], -1
            )
        hydra_loss, hydra_loss_dict = hydra_kd_imi_agent_loss(targets,
                                                              hydra_eval_results,
                                                              hydra_config,
                                                              hydra_gt_scores,
                                                              regression_ep=False,
                                                              three2two=True,
                                                              include_dp=True)

        loss += hydra_loss
        loss_dict.update(hydra_loss_dict)

    return loss, loss_dict


class DPAgent(AbstractAgent):
    def __init__(
            self,
            config: DPConfig,
            lr: float,
            checkpoint_path: str = None,
            pdm_gt_path=None,
            hydra_config: HydraConfig = None,
            hydra_checkpoint_path: str = None,
            scorer: PDMScorer = None,
            simulator: PDMSimulator = None,
            proposal_sampling: TrajectorySampling = None,
            reactive_agents_policy=None
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        if config.version == 'v1':
            self.model = DPModel(config)
        elif config.version == 'v2':
            raise ValueError('Unsupported')
        elif config.version == 'traj_v1':
            self.model = DPModel_traj_v1(config)
        elif config.version == 'traj_diffy':
            raise ValueError('Unsupported')
        elif config.version == 'traj_v1_bev':
            self.model = DPModel_traj_v1_bev(config)
        elif config.version == 'traj_v2_bev':
            self.model = DPModel_traj_v2_bev(config)
        elif config.version == 'traj_v2_bev_temporal':
            raise ValueError('Unsupported')
        elif config.version == 'traj_v2_bev_q':
            self.model = DPModel_traj_v2_bev_q(config)
        elif config.version == 'traj_v1_bev_ddpmft':
            self.model = DPModel_traj_v1_bev_ddpmft(config)
        elif config.version == 'traj_v2_bev_ddpmft':
            self.model = DPModel_traj_v2_bev_ddpmft(config)
        elif config.version == 'traj_v2_bev_ddpmft_orifirst':
            raise ValueError('Unsupported')
        elif config.version == 'traj_v1_bev_ddim':
            raise ValueError('Unsupported')
        else:
            raise ValueError('unsupported version')
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler
        self.version = config.version
        if 'ddpmft' in config.version:
            assert config.reward_source in ['hydra', 'pdm']
            self.hydra_model = HydraModel(hydra_config)
            self.hydra_config = hydra_config
            self.hydra_checkpoint_path = hydra_checkpoint_path
            self.hydra_config.trajectory_pdm_weight = {
                'no_at_fault_collisions': 3.0,
                'drivable_area_compliance': 3.0,
                'time_to_collision_within_bound': 4.0,
                'ego_progress': 2.0,
                'driving_direction_compliance': 1.0,
                'lane_keeping': 2.0,
                'traffic_light_compliance': 3.0,
                'history_comfort': 1.0,
            }
            self.metrics = list(self.hydra_config.trajectory_pdm_weight.keys())
            if pdm_gt_path is not None and self._config.open_hydra:
                self.vocab_pdm_score_full = pickle.load(
                    open(f'{os.getenv("NAVSIM_TRAJPDM_ROOT")}/{pdm_gt_path}', 'rb'))
            if config.reward_source == 'pdm':
                self.metric_cache_loader = MetricCacheLoader(Path(config.metric_cache_path))
                self.scorer = scorer
                self.simulator = simulator
                self.traffic_agents_policy_stage_one = reactive_agents_policy

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if not self._config.is_rl_training:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"]
            self.model.load_state_dict(
                {k.replace("agent.model.", ""): v for k, v in state_dict.items()}, strict=False
            )
            self.model._trajectory_head.update_reference_model()
            hydra_state_dict: Dict[str, Any] = torch.load(self.hydra_checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"]
            self.hydra_model.load_state_dict(
                {k.replace("agent.model.", ""): v for k, v in hydra_state_dict.items()}, strict=True
            )
            if hasattr(self.model._trajectory_head, 'ori_transformer'):
                transformer_dp_prefix = "agent.model._trajectory_head.transformer_dp."
                filtered_state_dict = {
                    k.replace(transformer_dp_prefix, ""): v
                    for k, v in state_dict.items()
                    if k.startswith(transformer_dp_prefix)
                }
                self.model._trajectory_head.ori_transformer.load_state_dict(
                    filtered_state_dict, strict=True
                )

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [HydraTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._config.is_rl_training:
            if self._config.hydra_filter:
                if self._config.open_hydra:
                    hydra_gt_scores = {}
                    for k in self.metrics:
                        tmp = [self.vocab_pdm_score_full[token][k][None] for token in tokens]
                        hydra_gt_scores[k] = (
                            torch.from_numpy(np.concatenate(tmp, axis=0)).to(predictions['bev_semantic_map'].device))
                else:
                    hydra_gt_scores = None
                return dp_loss_rl_filter(
                    targets, predictions, self._config, self.model._trajectory_head, self.hydra_model,
                    features,
                    metric_cache_loader=self.metric_cache_loader,
                    simulator=self.simulator,
                    scorer=self.scorer,
                    traffic_agents=self.traffic_agents_policy_stage_one,
                    tokens=tokens,
                    reinforce=self._config.reinforce,
                    hydra_config=self.hydra_config,
                    hydra_gt_scores=hydra_gt_scores
                )
            if self._config.reward_source == 'pdm':
                return dp_loss_rl(targets, predictions, self._config, self.model._trajectory_head, self.hydra_model,
                                  features,
                                  metric_cache_loader=self.metric_cache_loader,
                                  simulator=self.simulator,
                                  scorer=self.scorer,
                                  traffic_agents=self.traffic_agents_policy_stage_one,
                                  tokens=tokens,
                                  reinforce=self._config.reinforce)
            else:
                return dp_loss_rl(targets, predictions, self._config, self.model._trajectory_head, self.hydra_model,
                                  features,
                                  reinforce=self._config.reinforce)

        return dp_loss_bev(targets, predictions, self._config, self.model._trajectory_head)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        reference_transformer_name = '_trajectory_head.reference_transformer'
        ori_transformer_name = '_trajectory_head.ori_transformer'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.named_parameters()))
        default_params = list(filter(lambda kv:
                                     backbone_params_name not in kv[0] and
                                     reference_transformer_name not in kv[0] and
                                     ori_transformer_name not in kv[0], self.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]

        if self._config.open_hydra and hasattr(self, 'hydra_model'):
            hydra_default_params = list(filter(lambda kv:
                                         backbone_params_name not in kv[0], self.hydra_model.named_parameters()))
            hydra_img_backbone_params = list(filter(lambda kv: backbone_params_name in kv[0], self.hydra_model.named_parameters()))

            hydra_model_params = {'params': [tmp[1] for tmp in hydra_default_params]}
            hydra_backbone_params = {
                    'params': [tmp[1] for tmp in hydra_img_backbone_params],
                    'lr': self._lr * self.hydra_config.lr_mult_backbone,
                    'weight_decay': self.hydra_config.backbone_wd
                }
            params_lr_dict += [hydra_model_params, hydra_backbone_params]

        if self.scheduler == 'default':
            return torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.01,
                    total_steps=100 * 202
                )
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_callback = ModelCheckpoint(
            save_top_k=100,
            monitor="val/loss_epoch",
            mode="min",
            dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
            filename="{epoch:02d}-{step:04d}",
        )
        if 'ddpmft' in self.version:
            # ref model is always frozen
            return [
                ReferenceModelUpdateCallback(),
                ckpt_callback
            ]
        return [
            ckpt_callback
        ]
