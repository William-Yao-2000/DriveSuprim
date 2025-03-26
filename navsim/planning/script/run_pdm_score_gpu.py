import logging
import os
import pickle
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score import calculate_two_frame_extended_comfort, compute_final_scores, \
    infer_two_stage_mapping, validate_two_stage_mapping, calculate_pseudo_closed_loop_weights, \
    calculate_individual_mapping_scores
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


# TODO gpu inference

def run_pdm_score_wo_inference(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = args[0]['model_trajectory']

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
            simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy, simulator.proposal_sampling
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        navsim_blobs_path=Path(cfg.navsim_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[pd.DataFrame] = []

    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]['trajectory']
            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False
        score_row["token"] = token

        pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # GPU INFERENCE
    # gpu inference
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader_inference = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        navsim_blobs_path=Path(cfg.navsim_blobs_path),
        scene_filter=scene_filter,
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        navsim_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

    assert len(dataset) == len(tokens_to_evaluate), f'dataloader: {len(dataset)}, tokens: {len(tokens_to_evaluate)}'

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(
        AgentLightningModule(
            agent=agent,
        ),
        dataloader,
        return_predictions=True
    )

    dist.barrier()
    all_predictions = [None for _ in range(dist.get_world_size())]

    if dist.is_initialized():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions.append(predictions)

    if dist.get_rank() != 0:
        return None

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    agent_ckpt_path = Path(cfg.agent.checkpoint_path).parent.absolute().__str__()
    ckpt_name = Path(cfg.agent.checkpoint_path).name.split('.')[0]
    pickle.dump(merged_predictions, open(f'{agent_ckpt_path}/{ckpt_name}.pkl', 'wb'))

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score_wo_inference, data_points)

    pdm_score_df = pd.concat(score_rows)
    old_pdms = (pdm_score_df['no_at_fault_collisions'] *
                pdm_score_df['drivable_area_compliance'] *
                ((5 * pdm_score_df['ego_progress'] +
                  5 * pdm_score_df['time_to_collision_within_bound'] +
                  2 * pdm_score_df['history_comfort']) / 12))
    pdm_score_df['old_pdms'] = old_pdms
    # Calculate two-frame extended comfort
    two_frame_comfort_df = calculate_two_frame_extended_comfort(
        pdm_score_df, proposal_sampling=instantiate(cfg.simulator.proposal_sampling)
    )

    # Merge two-frame comfort scores and drop unnecessary columns in one step
    pdm_score_df = (
        pdm_score_df.drop(columns=["ego_simulated_states"])  # Remove the unwanted column first
        .merge(
            two_frame_comfort_df[["current_token", "two_frame_extended_comfort"]],
            left_on="token",
            right_on="current_token",
            how="left",
        )
        .drop(columns=["current_token"])  # Remove merged key after the merge
    )

    # Compute final scores
    pdm_score_df = compute_final_scores(pdm_score_df)

    try:
        if hasattr(cfg.train_test_split, "two_stage_mapping"):
            two_stage_mapping: Dict[str, List[str]] = dict(cfg.train_test_split.two_stage_mapping)
        else:
            # infer two stage mapping from results
            two_stage_mapping = infer_two_stage_mapping(pdm_score_df, first_stage_duration=4.0)
        validate_two_stage_mapping(pdm_score_df, two_stage_mapping)

        # calculate weights for pseudo closed loop using config
        weights = calculate_pseudo_closed_loop_weights(pdm_score_df, two_stage_mapping=two_stage_mapping)
        assert len(weights) == len(pdm_score_df), "Couldn't calculate weights for all tokens."
        pdm_score_df = pdm_score_df.merge(weights, on="token")
        pseudo_closed_loop_valid = True
    except Exception:
        logger.warning("----------- Failed to calculate pseudo closed-loop weights:")
        traceback.print_exc()
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = pdm_score_df[not pdm_score_df["valid"]]["token"].to_list()
    else:
        failed_tokens = []

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
                (any(score.name in c for score in
                     fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score" or c == 'old_pdms')
                and c != "pdm_score"
        )
    ]

    # Calculate average score
    average_row = pdm_score_df[score_cols].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = pdm_score_df["valid"].all()

    # Calculate pseudo closed loop score with weighted average
    pseudo_closed_loop_row = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], two_stage_mapping
    )
    pseudo_closed_loop_row["token"] = "pseudo_closed_loop"
    pseudo_closed_loop_row["valid"] = pseudo_closed_loop_valid

    # Original frames average
    original_frames = pdm_score_df[pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL]
    average_original_row = original_frames[score_cols].mean(skipna=True)
    average_original_row["token"] = "average_expert_frames"
    average_original_row["valid"] = original_frames["valid"].all()

    # append average and pseudo closed loop scores
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = average_row
    pdm_score_df.loc[len(pdm_score_df)] = pseudo_closed_loop_row
    pdm_score_df.loc[len(pdm_score_df)] = average_original_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Final old PDMS: {pdm_score_df['old_pdms'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":
    main()
