from typing import Dict, Optional, List
import os

import torch
import torch.nn.functional as F

from navsim.agents.hydra_ssl_v1.hydra_config_ssl import HydraConfigSSL


def hydra_kd_imi_agent_loss_robust(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: HydraConfigSSL,
        vocab_pdm_score
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    noc, da, ttc, comfort, progress = (predictions['noc'], predictions['da'],
                                       predictions['ttc'],
                                       predictions['comfort'], predictions['progress'])
    imi = predictions['imi']
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()
    # 2 cls
    if not config.only_imi_learning:
        da_loss = F.binary_cross_entropy(da, vocab_pdm_score['da'].to(da.dtype))
        ttc_loss = F.binary_cross_entropy(ttc, vocab_pdm_score['ttc'].to(da.dtype))
        comfort_loss = F.binary_cross_entropy(comfort, vocab_pdm_score['comfort'].to(da.dtype))
        noc_loss = F.binary_cross_entropy(noc, three_to_two_classes(vocab_pdm_score['noc'].to(da.dtype)))
        progress_loss = F.binary_cross_entropy(progress, vocab_pdm_score['progress'].to(progress.dtype))

    vocab = predictions["trajectory_vocab"]
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    # 4, 9, ..., 39
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
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()
    if config.weakly_supervised_imi_learning:
        weakly_selected_mark = targets['use_human_target'].squeeze(1)
        imi_se = imi[weakly_selected_mark]
        l2_distance_se = l2_distance[weakly_selected_mark]
        if imi.numel() == 0:
            imi_loss = imi.sum() * 0.0
        else:
            imi_loss = F.cross_entropy(imi_se, l2_distance_se.sum((-2, -1)).softmax(1))
    else:
        imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    imi_loss_final = config.trajectory_imi_weight * imi_loss

    if not config.only_imi_learning:
        noc_loss_final = config.trajectory_pdm_weight['noc'] * noc_loss
        da_loss_final = config.trajectory_pdm_weight['da'] * da_loss
        ttc_loss_final = config.trajectory_pdm_weight['ttc'] * ttc_loss
        progress_loss_final = config.trajectory_pdm_weight['progress'] * progress_loss
        comfort_loss_final = config.trajectory_pdm_weight['comfort'] * comfort_loss

    # agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)

    # agent_class_loss_final = config.agent_class_weight * agent_class_loss
    # agent_box_loss_final = config.agent_box_weight * agent_box_loss
    if config.only_imi_learning:
        loss = imi_loss_final
        return loss, {'imi_loss': imi_loss_final}
    else:
        loss = (
                imi_loss_final
                + noc_loss_final
                + da_loss_final
                + ttc_loss_final
                + progress_loss_final
                + comfort_loss_final
                # + agent_class_loss_final
                # + agent_box_loss_final
        )
        return loss, {
            'imi_loss': imi_loss_final,
            'pdm_noc_loss': noc_loss_final,
            'pdm_da_loss': da_loss_final,
            'pdm_ttc_loss': ttc_loss_final,
            'pdm_progress_loss': progress_loss_final,
            'pdm_comfort_loss': comfort_loss_final,
            # 'agent_class_loss': agent_class_loss_final,
            # 'agent_box_loss': agent_box_loss_final,
        }


def hydra_kd_imi_agent_loss_single_stage(
        predictions: Dict[str, torch.Tensor], config: HydraConfigSSL, vocab_pdm_score
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()
    layer_results = predictions['layer_results']
    losses = {}
    total_loss = 0.0
    for layer, layer_result in enumerate(layer_results):
        noc, da, ttc, comfort, progress = (layer_result['noc'], layer_result['da'],
                                           layer_result['ttc'],
                                           layer_result['comfort'], layer_result['progress'])

        da_loss = F.binary_cross_entropy(da, vocab_pdm_score['da'].to(da.dtype))
        ttc_loss = F.binary_cross_entropy(ttc, vocab_pdm_score['ttc'].to(da.dtype))
        comfort_loss = F.binary_cross_entropy(comfort, vocab_pdm_score['comfort'].to(da.dtype))
        noc_loss = F.binary_cross_entropy(noc, three_to_two_classes(vocab_pdm_score['noc'].to(da.dtype)))
        progress_loss = F.binary_cross_entropy(progress, vocab_pdm_score['progress'].to(progress.dtype))

        # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
        #     import pdb; pdb.set_trace()

        noc_loss_final = config.trajectory_pdm_weight['noc'] * noc_loss
        da_loss_final = config.trajectory_pdm_weight['da'] * da_loss
        ttc_loss_final = config.trajectory_pdm_weight['ttc'] * ttc_loss
        progress_loss_final = config.trajectory_pdm_weight['progress'] * progress_loss
        comfort_loss_final = config.trajectory_pdm_weight['comfort'] * comfort_loss

        loss = (noc_loss_final
                + da_loss_final
                + ttc_loss_final
                + progress_loss_final
                + comfort_loss_final
        )
        total_loss += loss
        losses[f'layer_{layer+1}'] = loss

    return total_loss, losses


def three_to_two_classes(x):
    x[x==0.5] = 0.0
    return x
