from typing import Dict, Optional, List
import os

import torch
import torch.nn.functional as F

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig


def bce_loss_with_temperature(
        predictions: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Binary Cross Entropy loss with temperature scaling.
    :param predictions: model predictions (logits)
    :param targets: ground truth labels
    :param temperature: temperature scaling factor
    :return: computed loss value
    """
    if temperature != 1.0:
        predictions = predictions / temperature
    return F.binary_cross_entropy_with_logits(predictions, targets)


def bce_loss_with_label_smoothing(
        predictions: torch.Tensor, targets: torch.Tensor, label_smoothing_value: float = 0.1,
) -> torch.Tensor:
    """
    Binary Cross Entropy loss with label smoothing.
    :param predictions: model predictions (logits)
    :param targets: ground truth labels
    :param label_smoothing_value: value for label smoothing
    :return: computed loss value
    """
    smooth_targets = targets * (1 - label_smoothing_value) + 0.5 * label_smoothing_value
    smooth_targets = smooth_targets.clamp(min=1e-7, max=1 - 1e-7)
    return F.binary_cross_entropy_with_logits(predictions, smooth_targets, reduction='mean')



def drivesuprim_agent_loss_first_stage(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DriveSuprimConfig,
        vocab_pdm_score
):
    """
    Helper function calculating loss of DriveSuprim first stage (coarse filtering)
    """
    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        import pdb; pdb.set_trace()
    
    bce_loss_fn = F.binary_cross_entropy_with_logits

    no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress = (
        predictions['no_at_fault_collisions'],
        predictions['drivable_area_compliance'],
        predictions['time_to_collision_within_bound'],
        predictions['ego_progress']
    )
    driving_direction_compliance, lane_keeping, traffic_light_compliance = (
        predictions['driving_direction_compliance'],
        predictions['lane_keeping'],
        predictions['traffic_light_compliance']
    )
    history_comfort = predictions['history_comfort']
    imi = predictions['imi']
    _dtype = imi.dtype

    da_loss = bce_loss_fn(drivable_area_compliance, vocab_pdm_score['drivable_area_compliance'].to(_dtype))
    ttc_loss = bce_loss_fn(time_to_collision_within_bound, vocab_pdm_score['time_to_collision_within_bound'].to(_dtype))
    noc_loss = bce_loss_fn(no_at_fault_collisions, three_to_two_classes(vocab_pdm_score['no_at_fault_collisions'].to(_dtype)))
    progress_loss = bce_loss_fn(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
    ddc_loss = bce_loss_fn(driving_direction_compliance, three_to_two_classes(vocab_pdm_score['driving_direction_compliance'].to(_dtype)))
    lk_loss = bce_loss_fn(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))
    tl_loss = bce_loss_fn(traffic_light_compliance,vocab_pdm_score['traffic_light_compliance'].to(_dtype))
    comfort_loss = bce_loss_fn(history_comfort,vocab_pdm_score['history_comfort'].to(_dtype))

    vocab = predictions["trajectory_vocab"]
    target_traj = targets["trajectory"]
    sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
    B = target_traj.shape[0]
    l2_distance = -((vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1) - target_traj[:, None]) ** 2) / config.sigma
    """
    vocab: [vocab_size, 40, 3]
    vocab[:, sampled_timepoints]: [vocab_size, 8, 3]
    vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1): [b, vocab_size, 8, 3]
    target_traj[:, None]: [b, 1, 8, 3]
    l2_distance: [b, vocab_size, 8, 3]
    """
    imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    imi_loss_final = config.trajectory_imi_weight * imi_loss

    noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
    da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
    ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
    progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
    ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
    lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
    tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
    comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

    loss = (
        imi_loss_final
        + noc_loss_final
        + da_loss_final
        + ttc_loss_final
        + progress_loss_final
        + ddc_loss_final
        + lk_loss_final
        + tl_loss_final
        + comfort_loss_final
    )
    return loss, {
        'imi_loss': imi_loss_final,
        'pdm_noc_loss': noc_loss_final,
        'pdm_da_loss': da_loss_final,
        'pdm_ttc_loss': ttc_loss_final,
        'pdm_progress_loss': progress_loss_final,
        'pdm_ddc_loss': ddc_loss_final,
        'pdm_lk_loss': lk_loss_final,
        'pdm_tl_loss': tl_loss_final,
        'pdm_comfort_loss': comfort_loss_final
    }


def drivesuprim_agent_loss_single_refine_stage(
        predictions: Dict[str, torch.Tensor], config: DriveSuprimConfig, vocab_pdm_score, targets=None
):
    """
    Helper function calculating loss of single refinement stage of DriveSuprim
    """

    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        import pdb; pdb.set_trace()

    bce_loss_fn = F.binary_cross_entropy_with_logits

    layer_results = predictions['layer_results']
    losses = {}
    total_loss = 0.0
    
    for layer, layer_result in enumerate(layer_results):
        
        no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress = (
            layer_result['no_at_fault_collisions'],
            layer_result['drivable_area_compliance'],
            layer_result['time_to_collision_within_bound'],
            layer_result['ego_progress']
        )
        driving_direction_compliance, lane_keeping, traffic_light_compliance = (
            layer_result['driving_direction_compliance'],
            layer_result['lane_keeping'],
            layer_result['traffic_light_compliance']
        )
        history_comfort = layer_result['history_comfort']
    
        _dtype = drivable_area_compliance.dtype

        da_loss = bce_loss_fn(drivable_area_compliance, vocab_pdm_score['drivable_area_compliance'].to(_dtype))
        ttc_loss = bce_loss_fn(time_to_collision_within_bound, vocab_pdm_score['time_to_collision_within_bound'].to(_dtype))
        noc_loss = bce_loss_fn(no_at_fault_collisions, three_to_two_classes(vocab_pdm_score['no_at_fault_collisions'].to(_dtype)))
        progress_loss = bce_loss_fn(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
        ddc_loss = bce_loss_fn(driving_direction_compliance, three_to_two_classes(vocab_pdm_score['driving_direction_compliance'].to(_dtype)))
        lk_loss = bce_loss_fn(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))
        tl_loss = bce_loss_fn(traffic_light_compliance, vocab_pdm_score['traffic_light_compliance'].to(_dtype))
        comfort_loss = bce_loss_fn(history_comfort, vocab_pdm_score['history_comfort'].to(_dtype))

        noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
        da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
        ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
        progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
        ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
        lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
        tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
        comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

        loss = (
            noc_loss_final
            + da_loss_final
            + ttc_loss_final
            + progress_loss_final
            + ddc_loss_final
            + lk_loss_final
            + tl_loss_final
            + comfort_loss_final
        )

        if config.refinement.use_imi_learning_in_refinement:
            imi = layer_result['imi']
            vocab = predictions["trajectory_vocab"]
            target_traj = targets["trajectory"]
            sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
            indices_absolute = predictions['indices_absolute']
            l2_distance = -((vocab[:, sampled_timepoints][indices_absolute] - target_traj[:, None]) ** 2) / config.sigma

            imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))
            imi_loss_final = config.trajectory_imi_weight * imi_loss
            loss += imi_loss_final
        
        total_loss += loss
        losses[f'layer_{layer+1}'] = loss

    return total_loss, losses


def three_to_two_classes(x):
    x[x==0.5] = 0.0
    return x
