import logging
import os
import pickle

import numpy as np
import torch

logger = logging.getLogger(__name__)

"""
pkl -> search params and calculation process
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pkl_path', required=True)


def linspace(start, end, cnt):
    return list(np.linspace(start, end, num=(cnt + 1)))

class PDMScorerConfigExpanded:
    # weighted metric weights
    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    comfortable_weight: float = 5.0
    lk_weight: float = 5.0

    # thresholds
    driving_direction_horizon: float = 1.0  # [s] (driving direction)
    driving_direction_compliance_threshold: float = 0.5  # [m] (driving direction)
    driving_direction_violation_threshold: float = 1.5  # [m] (driving direction)
    stopped_speed_threshold: float = 5e-03  # [m/s] (ttc)
    progress_distance_threshold: float = 0.1  # [m] (progress)

    @property
    def weighted_metrics_array(self) :
        weighted_metrics = np.zeros(4, dtype=np.float64)
        weighted_metrics[0] = self.progress_weight
        weighted_metrics[1] = self.ttc_weight
        weighted_metrics[2] = self.comfortable_weight
        # weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = self.lk_weight
        return weighted_metrics

    @property
    def weighted_metrics_array_expand(self) :
        weighted_metrics = np.zeros(4, dtype=np.float64)
        weighted_metrics[0] = self.progress_weight
        weighted_metrics[1] = self.ttc_weight
        weighted_metrics[2] = self.comfortable_weight
        weighted_metrics[3] = self.lk_weight
        return weighted_metrics


def main() -> None:
    args = parser.parse_args()
    pkl_path = args.pkl_path

    merged_predictions = pickle.load(open(pkl_path, 'rb'))
    navtest_scores = pickle.load(
        open(f'{os.getenv("NAVSIM_TRAJPDM_ROOT")}/vocab_score_full_8192_navtest/navtest.pkl', 'rb')
    )
    navtest_scores_expand = pickle.load(
        open(f'{os.getenv("NAVSIM_TRAJPDM_ROOT")}/vocab_expanded_8192_navtest/navtest.pkl', 'rb')
    )
    navtest_scores_newlk = pickle.load(
        open(f'{os.getenv("NAVSIM_TRAJPDM_ROOT")}/vocab_newlk_8192_navtest/navtest.pkl', 'rb')
    )
    # print(navtest_scores.keys())
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
    #         0.5 * result['noc'].log() +
    #         0.5 * result['da'].log() +
    #         8.0 * (5 * result['ttc'] + 2 * result['comfort'] + 5 * result['progress']).log()
    # )
    # temporary
    # imi_weights = [0.01 * tmp for tmp in range(1, 101)]
    # noc_weights = [0.1 * tmp for tmp in range(1, 11)]
    # da_weights = [0.1 * tmp for tmp in range(1, 11)]
    # tpc_weights = [1.0 * tmp for tmp in range(1, 11)]
    # ttc_weights = [5.0]
    # progress_weights = [5.0]
    # comfort_weights = [2.0]

    # imi_weights = [0.0025 * tmp for tmp in range(2, 6)]
    imi_weights = [0.01]
    # noc_weights = [0.0025 * tmp for tmp in range(3, 6)]
    noc_weights = [0.01]
    # da_weights = [0.0025 * tmp for tmp in range(3, 6)]
    da_weights = [0.0025 * tmp for tmp in range(3, 6)]
    # lk_weights = [0.0125 * tmp for tmp in range(3, 7)]
    lk_weights = [0.5 * tmp for tmp in range(1, 10)]
    # dd_weights = [0.0125 * tmp for tmp in range(3, 7)]
    dd_weights = [0.0125 * tmp for tmp in range(3, 7)]
    tl_weights = [1.0 * tmp for tmp in range(1, 10)]
    tpc_weights = [3.5]
    ttc_weights = [7.0]
    progress_weights = [1.0 * tmp for tmp in range(3, 6)]
    comfort_weights = [1.0]

    # imi_weights = [0.01]
    # noc_weights = [0.01]
    # da_weights = [0.01]
    # tpc_weights = [3.5]
    # ttc_weights = [7.0]
    # progress_weights = [4.0]
    # comfort_weights = [1.0]
    # tl_weights = [6.0]
    # lk_weights = [3.0]
    # dd_weights = [0.075]
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
    pdm_scores, noc_scores, da_scores, dd_scores, ttc_scores, progress_scores, comfort_scores, lk_scores, tl_scores= (
    [], [], [], [], [], [], [], [], [])
    total_scene_cnt = len(navtest_scores)
    print(f'total_scene_cnt: {total_scene_cnt}')
    for k, v in navtest_scores.items():
        vv = navtest_scores_expand[k]
        vvv = navtest_scores_newlk[k]

        # pdm_scores.append(torch.from_numpy(v['total_expand'][None]).cuda())
        noc_scores.append(torch.from_numpy(v['noc'][None]).cuda())
        da_scores.append(torch.from_numpy(v['da'][None]).cuda())
        dd_scores.append(torch.from_numpy(vv['dr'][None]).cuda())
        ttc_scores.append(torch.from_numpy(v['ttc'][None]).cuda())
        progress_scores.append(torch.from_numpy(v['progress'][None]).cuda())
        # comfort_scores.append(torch.from_numpy(merged_predictions[k]['comfort'][None]).cuda())
        lk_scores.append(torch.from_numpy(vvv['lk'][None]).cuda())
        tl_scores.append(torch.from_numpy(vv['tl'][None]).cuda())

        imi_preds.append(torch.from_numpy(merged_predictions[k]['imi'][None]).cuda())
        noc_preds.append(torch.from_numpy(merged_predictions[k]['noc'][None]).cuda())
        da_preds.append(torch.from_numpy(merged_predictions[k]['da'][None]).cuda())
        ttc_preds.append(torch.from_numpy(merged_predictions[k]['ttc'][None]).cuda())
        progress_preds.append(torch.from_numpy(merged_predictions[k]['progress'][None]).cuda())
        # comfort_preds.append(torch.from_numpy(merged_predictions[k]['comfort'][None]).cuda())
        lk_preds.append(torch.from_numpy(merged_predictions[k]['lk'][None]).cuda())
        tl_preds.append(torch.from_numpy(merged_predictions[k]['tl'][None]).cuda())
        dd_preds.append(torch.from_numpy(merged_predictions[k]['dr'][None]).cuda())

        config = PDMScorerConfigExpanded()
        weighted_metrics_array = config.weighted_metrics_array_expand
        _weighted_metrics = np.zeros(
            (4, 8192), dtype=np.float64
        )
        _multi_metrics = np.zeros(
            (4, 8192), dtype=np.float64
        )
        # print(v['progress'], v['ttc'])
        _weighted_metrics[0] = v['progress']
        _weighted_metrics[1] = v['ttc']
        _weighted_metrics[2] = np.full_like(v['progress'], 0.9683)
        _weighted_metrics[3] = vvv['lk']

        _multi_metrics[0] = v['noc']
        _multi_metrics[1] = v['da']
        _multi_metrics[2] = vv['dr']
        _multi_metrics[3] = vv['tl']

        weighted_metric_scores = (_weighted_metrics * weighted_metrics_array[..., None]).sum(
            axis=0
        )
        weighted_metric_scores /= weighted_metrics_array.sum()
        # _multi_metrics = np.nan_to_num(_multi_metrics, nan=0.0, posinf=0.0, neginf=0.0)
        # weighted_metric_scores = np.nan_to_num(weighted_metric_scores, nan=0.0, posinf=0.0, neginf=0.0)
        # calculate final scores
        final_scores = _multi_metrics.prod(axis=0) * weighted_metric_scores
        pdm_scores.append(torch.from_numpy(final_scores[None]).cuda())



    pdm_scores = torch.cat(pdm_scores, 0).contiguous()
    noc_scores = torch.cat(noc_scores, 0).contiguous()
    da_scores = torch.cat(da_scores, 0).contiguous()
    # dd_scores = torch.cat(dd_scores, 0).contiguous()
    ttc_scores = torch.cat(ttc_scores, 0).contiguous()
    progress_scores = torch.cat(progress_scores, 0).contiguous()
    # comfort_scores = torch.cat(comfort_scores, 0).contiguous()
    lk_scores = torch.cat(lk_scores, 0).contiguous()
    tl_scores = torch.cat(tl_scores, 0).contiguous()
    dd_scores = torch.cat(dd_scores, 0).contiguous()

    imi_preds = torch.cat(imi_preds, 0).contiguous()
    noc_preds = torch.cat(noc_preds, 0).contiguous()
    da_preds = torch.cat(da_preds, 0).contiguous()
    ttc_preds = torch.cat(ttc_preds, 0).contiguous()
    progress_preds = torch.cat(progress_preds, 0).contiguous()
    # comfort_preds = torch.cat(comfort_preds, 0).contiguous()
    lk_preds = torch.cat(lk_preds, 0).contiguous()
    tl_preds = torch.cat(tl_preds, 0).contiguous()
    dd_preds = torch.cat(dd_preds, 0).contiguous()

    rows = []
    highest_info = {
        'score': -100,
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
                                                            # comfort_weight * torch.exp(comfort_preds) +
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
                                            # comfort_score = comfort_scores[
                                            #     scene_cnt_tensor,
                                            #     chosen_idx
                                            # ]
                                            lk_score = lk_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]
                                            tl_score = tl_scores[
                                                scene_cnt_tensor,
                                                chosen_idx
                                            ]

                                            pdm_score = pdm_score.mean().item()
                                            noc_score = noc_score.float().mean().item()
                                            da_score = da_score.float().mean().item()
                                            dd_score = dd_score.float().mean().item()
                                            lk_score = lk_score.float().mean().item()
                                            tl_score = tl_score.float().mean().item()
                                            ttc_score = ttc_score.float().mean().item()
                                            progress_score = progress_score.float().mean().item()
                                            # comfort_score = comfort_score.float().mean().item()
                                            row = {
                                                'imi_weight': imi_weight,
                                                'noc_weight': noc_weight,
                                                'da_weight': da_weight,
                                                'ttc_weight': ttc_weight,
                                                'progress_weight': progress_weight,
                                                'comfort_weight': comfort_weight,
                                                'tpc_weight': tpc_weight,
                                                'lk_weight': lk_weight,
                                                'tl_weight': tl_weight,
                                                'dd_weight': dd_weight,
                                                'overall_score': pdm_score
                                            }
                                            if pdm_score > highest_info['score']:
                                                highest_info['score'] = pdm_score
                                                highest_info['noc'] = noc_score
                                                highest_info['da'] = da_score
                                                highest_info['dd'] = dd_score
                                                highest_info['ttc'] = ttc_score
                                                highest_info['lk'] = lk_score
                                                highest_info['tl'] = tl_score
                                                highest_info['progress'] = progress_score
                                                # highest_info['comfort'] = comfort_score
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
                                            print(f'Done: {len(rows)}. score: {pdm_score}')
                                            rows.append(row)
    # save rows
    # pdm_score_df = pd.DataFrame(rows)
    # pdm_score_df.to_csv(Path(csv_path))
    for k, v in highest_info.items():
        print(k, v)


if __name__ == "__main__":
    with torch.no_grad():
        main()
