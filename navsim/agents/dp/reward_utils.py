from concurrent import futures

import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.hydra_plus.hydra_features import state2traj
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


def hydra_eval_dp(features, dp_preds, hydra_model, v_params=get_pacifica_parameters(), open_hydra=False, topk=10):
    device = dp_preds.device
    B = dp_preds.shape[0]
    all_interpolated_proposals = []
    for batch_idx in range(B):
        proposals = dp_preds[batch_idx]
        ego_state = EgoState.build_from_rear_axle(
            StateSE2(*features['ego_pose'].cpu().numpy()[batch_idx]),
            tire_steering_angle=0.0,
            vehicle_parameters=v_params,
            time_point=TimePoint(0),
            rear_axle_velocity_2d=StateVector2D(
                *features['ego_velocity'].cpu().numpy()[batch_idx]
            ),
            rear_axle_acceleration_2d=StateVector2D(
                *features['ego_acceleration'].cpu().numpy()[batch_idx]
            ),
        )
        interpolated_proposals = []
        for proposal in proposals:
            traj = Trajectory(
                proposal.detach().cpu(),
                TrajectorySampling(time_horizon=4, interval_length=0.5)
            )
            trans_traj = transform_trajectory(
                traj, ego_state
            )
            interpolated_traj = get_trajectory_as_array(
                trans_traj,
                TrajectorySampling(num_poses=40, interval_length=0.1),
                ego_state.time_point
            )
            final_traj = state2traj(interpolated_traj)
            interpolated_proposals.append(final_traj)
        interpolated_proposals = np.array(interpolated_proposals)
        interpolated_proposals = torch.from_numpy(interpolated_proposals).float().to(device)[None]
        all_interpolated_proposals.append(interpolated_proposals)
    # B, 100, 40, 3
    all_interpolated_proposals = torch.cat(all_interpolated_proposals, 0)
    if open_hydra:
        predictions = hydra_model.evaluate_dp_proposals(features, all_interpolated_proposals, topk=topk)
    else:
        with torch.no_grad():
            predictions = hydra_model.evaluate_dp_proposals(features, all_interpolated_proposals, topk=topk)
    return predictions


def pdm_eval_dp_singlethread(tokens, proposals, metric_cache_loader,
                             simulator, scorer, traffic_agents_policy_stage_one):
    pdms_all = []
    for token, proposal in zip(tokens, proposals):
        metric_cache = metric_cache_loader.get_from_token(token)
        pdms, ego_states_all = pdm_eval_dp_single(
            proposal.cpu().numpy(), metric_cache, simulator, scorer, traffic_agents_policy_stage_one
        )
        pdms_all.append(torch.tensor(pdms['pdm_score'], device=proposal.device).float()[None])
    return torch.cat(pdms_all, 0)


def pdm_eval_dp_multithread(tokens, proposals, metric_cache_loader,
                            simulator, scorer, traffic_agents_policy_stage_one):
    # 创建独立的对象池（每个线程一个实例）
    def create_worker_objects():
        return {
            "simulator": PDMSimulator(
                proposal_sampling=TrajectorySampling(num_poses=simulator.proposal_sampling.num_poses,
                                                     interval_length=simulator.proposal_sampling.interval_length)),
            "scorer": PDMScorer(
                config=scorer._config,
                proposal_sampling=TrajectorySampling(num_poses=scorer.proposal_sampling.num_poses,
                                                     interval_length=scorer.proposal_sampling.interval_length)
            ),
            "traffic_agents": type(traffic_agents_policy_stage_one)(future_trajectory_sampling=TrajectorySampling(
                num_poses=traffic_agents_policy_stage_one.future_trajectory_sampling.num_poses,
                interval_length=traffic_agents_policy_stage_one.future_trajectory_sampling.interval_length
            ))
        }

    # 预创建所有工作对象
    worker_objects = [create_worker_objects() for _ in range(len(tokens))]

    # 准备参数列表（包含独立对象）
    args_list = [
        (token, proposal,
         metric_cache_loader.get_from_token(token),
         worker_objects[i]["simulator"],
         worker_objects[i]["scorer"],
         worker_objects[i]["traffic_agents"])
        for i, (token, proposal) in enumerate(zip(tokens, proposals))
    ]

    # 处理函数保持不变
    def process_one(args):
        token, proposal, metric_cache, simulator, scorer, traffic_agents = args
        pdms, _ = pdm_eval_dp_single(
            proposal.cpu().numpy(), metric_cache, simulator, scorer, traffic_agents
        )
        return pdms
        # return torch.tensor(pdms['pdm_score'], device=proposal.device).float()[None]

    # 使用线程池处理
    with futures.ThreadPoolExecutor(max_workers=len(args_list)) as executor:
        pdms_all = list(executor.map(process_one, args_list))

    result = {}
    for k in pdms_all[0]:
        scores = torch.tensor([subscore[k] for subscore in pdms_all], device=proposals.device).float()
        result[k] = scores
    return result


def pdm_eval_dp_single(proposals,
                       metric_cache,
                       simulator,
                       scorer,
                       traffic_agents_policy_stage_one):
    """
    :param proposals: np.array: N, 8, 3
    :return:
    """
    if proposals.shape[1] == 40:
        interval_length = 0.1
    else:
        interval_length = 0.5
    scores = {
        'no_at_fault_collisions': [],
        'drivable_area_compliance': [],
        'driving_direction_compliance': [],
        'traffic_light_compliance': [],
        'ego_progress': [],
        'time_to_collision_within_bound': [],
        'lane_keeping': [],
        'history_comfort': [],
        'pdm_score': []
    }
    ego_states_all = []
    for proposal in proposals:
        traj = Trajectory(
            proposal,
            TrajectorySampling(time_horizon=4,
                               interval_length=interval_length)
        )
        score_row, ego_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=traj,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy_stage_one,
        )
        ego_states_all.append(ego_states)
        for metric in list(score_row.keys()):
            if metric in scores:
                scores[metric].append(score_row[metric][0])
    return scores, ego_states_all
