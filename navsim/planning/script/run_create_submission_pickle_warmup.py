import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module_ssl import AgentLightningModuleSSL
from navsim.planning.training.dataset_ssl import DatasetSSL as Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"


def run_test_evaluation(
    cfg: DictConfig,
    agent: AbstractAgent,
    scene_filter: SceneFilter,
    data_path: Path,
    synthetic_sensor_path: Path,
    original_sensor_path: Path,
    synthetic_scenes_path: Path,
) -> Dict[str, Trajectory]:
    """
    Function to create the output file for evaluation of an agent on the testserver
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param synthetic_sensor_path: pathlib path to sensor blobs
    :param synthetic_scenes_path: pathlib path to synthetic scenes
    :param save_path: pathlib path to folder where scores are stored as .csv
    """
    if agent.requires_scene:
        raise ValueError(
            """
            In evaluation, no access to the annotated scene is provided, but only to the AgentInput.
            Thus, agent.requires_scene has to be False for the agent that is to be evaluated.
            """
        )
    logger.info("Building Agent Input Loader")
    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        synthetic_sensor_path=synthetic_sensor_path,
        original_sensor_path=original_sensor_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=agent.get_sensor_config(),
    )
    agent.initialize()

    # first stage output
    first_stage_output: Dict[str, Trajectory] = {}
    for token in tqdm(input_loader.tokens_stage_one, desc="Running first stage evaluation"):
        try:
            agent_input = input_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
            first_stage_output.update({token: trajectory})
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()

    # second stage output

    scene_loader_tokens_stage_two = input_loader.reactive_tokens_stage_two

    second_stage_output: Dict[str, Trajectory] = {}
    for token in tqdm(scene_loader_tokens_stage_two, desc="Running second stage evaluation"):
        try:
            agent_input = input_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
            second_stage_output.update({token: trajectory})
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()

    return first_stage_output, second_stage_output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """
    agent = instantiate(cfg.agent)
    agent.initialize()

    data_path = Path(cfg.navsim_log_path)
    synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader_inference = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cfg=cfg.agent.config,
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(
        AgentLightningModuleSSL(
            cfg=cfg.agent.config,
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

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader_inference.get_tokens_list_per_log().items()
    ]

    import pdb; pdb.set_trace()

    first_stage_output, second_stage_output = run_test_evaluation(
        cfg=cfg,
        agent=agent,
        scene_filter=scene_filter,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        synthetic_sensor_path=synthetic_sensor_path,
        original_sensor_path=original_sensor_path,
    )

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "first_stage_predictions": [first_stage_output],
        "second_stage_predictions": [second_stage_output],
    }

    # pickle and save dict
    filename = os.path.join(save_path, "submission.pkl")
    with open(filename, "wb") as file:
        pickle.dump(submission, file)
    logger.info(f"Your submission filed was saved to {filename}")


if __name__ == "__main__":
    main()
