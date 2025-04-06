import logging
import os
import pickle

import numpy as np
import torch
import tqdm

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

logger = logging.getLogger(__name__)

"""
pkl -> search params and calculation process
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pkl_path', required=True)


def linspace(start, end, cnt):
    return list(np.linspace(start, end, num=(cnt + 1)))


def aggregate_old_pdms(gt_score):
    weighted_metrics_array = np.zeros(3, dtype=np.float64)
    weighted_metrics_array[0] = 5.0
    weighted_metrics_array[1] = 5.0
    weighted_metrics_array[2] = 2.0
    _weighted_metrics = np.zeros(
        (3, 16384), dtype=np.float64
    )
    _multi_metrics = np.zeros(
        (2, 16384), dtype=np.float64
    )
    _weighted_metrics[0] = gt_score['ego_progress']
    _weighted_metrics[1] = gt_score['time_to_collision_within_bound']
    _weighted_metrics[2] = gt_score['history_comfort']

    _multi_metrics[0] = gt_score['no_at_fault_collisions']
    _multi_metrics[1] = gt_score['drivable_area_compliance']

    weighted_metric_scores = (_weighted_metrics * weighted_metrics_array[..., None]).sum(
        axis=0
    )
    weighted_metric_scores /= weighted_metrics_array.sum()
    # _multi_metrics = np.nan_to_num(_multi_metrics, nan=0.0, posinf=0.0, neginf=0.0)
    # weighted_metric_scores = np.nan_to_num(weighted_metric_scores, nan=0.0, posinf=0.0, neginf=0.0)
    # calculate final scores
    final_scores = _multi_metrics.prod(axis=0) * weighted_metric_scores
    return final_scores


def main() -> None:
    args = parser.parse_args()
    pkl_path = args.pkl_path

    merged_predictions = pickle.load(open(pkl_path, 'rb'))
    navtest_scores = pickle.load(
        open(f'{os.getenv("NAVSIM_TRAJPDM_ROOT")}/vocab_score_full_16384_navtest_v2ep/navtest.pkl', 'rb')
    )

    # standard
    # imi_weights = [0.01 * tmp for tmp in range(1, 11)]
    # noc_weights = [0.1 * tmp for tmp in range(1, 11)]
    # da_weights = [0.1 * tmp for tmp in range(1, 11)]
    # tpc_weights = [1.0 * tmp for tmp in range(1, 11)]
    # ttc_weights = [5.0]
    # progress_weights = [5.0]
    # comfort_weights = [2.0]
    # scores = (
    #         0.05 * result['imi'].softmax(-1).log() +
    #         0.5 * result['no_at_fault_collisions].log() +
    #         0.5 * result['drivable_area_compliance].log() +
    #         8.0 * (5 * result['time_to_collision_within_bound'] + 2 * result['history_comfort'] + 5 * result['ego_progress']).log()
    # )
    # temporary
    # imi_weights = [0.01 * tmp for tmp in range(1, 101)]
    # noc_weights = [0.1 * tmp for tmp in range(1, 11)]
    # da_weights = [0.1 * tmp for tmp in range(1, 11)]
    # tpc_weights = [1.0 * tmp for tmp in range(1, 11)]
    # ttc_weights = [5.0]
    # progress_weights = [5.0]
    # comfort_weights = [2.0]

    imi_weights = [0.02, 0.03, 0.04]
    noc_weights = [0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    da_weights = [0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    dd_weights = [0.2]
    tl_weights = [0.1]
    tpc_weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ttc_weights = [3.0]
    progress_weights = [6.0]
    lk_weights = [2.0]
    comfort_weights = [2.0]


    print(
        f'Search space: {len(imi_weights) * len(noc_weights) * len(da_weights) * len(tpc_weights) * len(ttc_weights) * len(progress_weights) * len(comfort_weights)}')

    (imi_preds,
     noc_preds,
     da_preds,
     dd_preds,
     ttc_preds,
     progress_preds,
     comfort_preds,
     lk_preds,
     tl_preds) = ([], [],
                  [], [],
                  [], [],
                  [], [], [])
    pdm_scores, noc_scores, da_scores, dd_scores, ttc_scores, progress_scores, comfort_scores, lk_scores, tl_scores = (
        [], [], [], [], [], [], [], [], [])
    old_pdm_scores = []
    total_scene_cnt = len(navtest_scores)
    print(f'total_scene_cnt: {total_scene_cnt}')
    for k, gt_score in tqdm.tqdm(navtest_scores.items()):
        pdm_scores.append(torch.from_numpy(gt_score['pdm_score'][None]).cuda())
        noc_scores.append(torch.from_numpy(gt_score['no_at_fault_collisions'][None]).cuda())
        da_scores.append(torch.from_numpy(gt_score['drivable_area_compliance'][None]).cuda())
        dd_scores.append(torch.from_numpy(gt_score['driving_direction_compliance'][None]).cuda())
        ttc_scores.append(torch.from_numpy(gt_score['time_to_collision_within_bound'][None]).cuda())
        progress_scores.append(torch.from_numpy(gt_score['ego_progress'][None]).cuda())
        comfort_scores.append(torch.from_numpy(merged_predictions[k]['history_comfort'][None]).cuda())
        lk_scores.append(torch.from_numpy(gt_score['lane_keeping'][None]).cuda())
        tl_scores.append(torch.from_numpy(gt_score['traffic_light_compliance'][None]).cuda())
        old_pdm_scores.append(torch.from_numpy(aggregate_old_pdms(
            gt_score
        )[None]).cuda())

        imi_preds.append(torch.from_numpy(merged_predictions[k]['imi'][None]).cuda())
        noc_preds.append(torch.from_numpy(merged_predictions[k]['no_at_fault_collisions'][None]).cuda())
        da_preds.append(torch.from_numpy(merged_predictions[k]['drivable_area_compliance'][None]).cuda())
        ttc_preds.append(torch.from_numpy(merged_predictions[k]['time_to_collision_within_bound'][None]).cuda())
        progress_preds.append(torch.from_numpy(merged_predictions[k]['ego_progress'][None]).cuda())
        comfort_preds.append(torch.from_numpy(merged_predictions[k]['history_comfort'][None]).cuda())
        lk_preds.append(torch.from_numpy(merged_predictions[k]['lane_keeping'][None]).cuda())
        tl_preds.append(torch.from_numpy(merged_predictions[k]['traffic_light_compliance'][None]).cuda())
        dd_preds.append(torch.from_numpy(merged_predictions[k]['driving_direction_compliance'][None]).cuda())


    pdm_scores = torch.cat(pdm_scores, 0).contiguous()
    old_pdm_scores = torch.cat(old_pdm_scores, 0).contiguous()
    noc_scores = torch.cat(noc_scores, 0).contiguous()
    da_scores = torch.cat(da_scores, 0).contiguous()
    ttc_scores = torch.cat(ttc_scores, 0).contiguous()
    progress_scores = torch.cat(progress_scores, 0).contiguous()
    comfort_scores = torch.cat(comfort_scores, 0).contiguous()
    lk_scores = torch.cat(lk_scores, 0).contiguous()
    tl_scores = torch.cat(tl_scores, 0).contiguous()
    dd_scores = torch.cat(dd_scores, 0).contiguous()

    imi_preds = torch.cat(imi_preds, 0).contiguous()
    noc_preds = torch.cat(noc_preds, 0).contiguous()
    da_preds = torch.cat(da_preds, 0).contiguous()
    ttc_preds = torch.cat(ttc_preds, 0).contiguous()
    progress_preds = torch.cat(progress_preds, 0).contiguous()
    comfort_preds = torch.cat(comfort_preds, 0).contiguous()
    lk_preds = torch.cat(lk_preds, 0).contiguous()
    tl_preds = torch.cat(tl_preds, 0).contiguous()
    dd_preds = torch.cat(dd_preds, 0).contiguous()

    search_cnt = 0
    highest_info = {
        'pdms': -100,
    }
    for lk_weight in lk_weights:
        for tl_weight in tl_weights:
            for dd_weight in dd_weights:
                for imi_weight in imi_weights:
                    for noc_weight in noc_weights:
                        for da_weight in da_weights:
                            for ttc_weight in ttc_weights:
                                for comfort_weight in comfort_weights:
                                    for progress_weight in progress_weights:
                                        for tpc_weight in tpc_weights:
                                            # old
                                            scores = (
                                                    imi_weight * imi_preds +
                                                    noc_weight * noc_preds +
                                                    da_weight * da_preds +
                                                    dd_weight * dd_preds +
                                                    tl_weight * tl_preds +
                                                    tpc_weight * (
                                                            ttc_weight * torch.exp(ttc_preds) +
                                                            comfort_weight * torch.exp(comfort_preds) +
                                                            progress_weight * torch.exp(progress_preds) +
                                                            lk_weight * torch.exp(lk_preds)
                                                    ).log()
                                            )
                                            chosen_idx = scores.argmax(-1)
                                            scene_cnt_tensor = torch.arange(total_scene_cnt, device=pdm_scores.device)
                                            pdm_score = pdm_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            old_pdm_score = old_pdm_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            noc_score = noc_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            da_score = da_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            dd_score = dd_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            ttc_score = ttc_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            progress_score = progress_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            comfort_score = comfort_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            lk_score = lk_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            tl_score = tl_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]

                                            pdm_score = pdm_score.mean().item()
                                            old_pdm_score = old_pdm_score.mean().item()
                                            noc_score = noc_score.float().mean().item()
                                            da_score = da_score.float().mean().item()
                                            dd_score = dd_score.float().mean().item()
                                            lk_score = lk_score.float().mean().item()
                                            tl_score = tl_score.float().mean().item()
                                            ttc_score = ttc_score.float().mean().item()
                                            progress_score = progress_score.float().mean().item()
                                            comfort_score = comfort_score.float().mean().item()

                                            if pdm_score > highest_info['pdms']:
                                                highest_info['pdms'] = pdm_score
                                                highest_info['old_pdms'] = old_pdm_score
                                                highest_info['no_at_fault_collisions'] = noc_score
                                                highest_info['drivable_area_compliance'] = da_score
                                                highest_info['driving_direction_compliance'] = dd_score
                                                highest_info['time_to_collision_within_bound'] = ttc_score
                                                highest_info['lane_keeping'] = lk_score
                                                highest_info['traffic_light_compliance'] = tl_score
                                                highest_info['ego_progress'] = progress_score
                                                highest_info['history_comfort'] = comfort_score
                                                highest_info['imi_weight'] = imi_weight
                                                highest_info['noc_weight'] = noc_weight
                                                highest_info['da_weight'] = da_weight
                                                highest_info['ttc_weight'] = ttc_weight
                                                highest_info['progress_weight'] = progress_weight
                                                highest_info['comfort_weight'] = comfort_weight
                                                highest_info['tpc_weight'] = tpc_weight
                                                highest_info['lk_weight'] = lk_weight
                                                highest_info['tl_weight'] = tl_weight
                                                highest_info['dd_weight'] = dd_weight
                                            search_cnt += 1
                                            print(f'Done: {search_cnt}. pdms: {pdm_score}. old_pdms: {old_pdm_score}')

    for k, gt_score in highest_info.items():
        print(k, gt_score)


if __name__ == "__main__":
    with torch.no_grad():
        main()
