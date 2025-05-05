import logging
import os
import pickle
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.common.dataloader_private import SceneLoader
from navsim.planning.training.agent_lightning_module_ssl import AgentLightningModuleSSL
from navsim.planning.training.dataset_ssl import DatasetSSL as Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """
    agent = instantiate(cfg.agent)
    agent.initialize()

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

    print(f'PKL Saved to {os.getenv("SUBSCORE_PATH")}')
    with open(os.getenv('SUBSCORE_PATH'), "wb") as file:
        pickle.dump(merged_predictions, file)


if __name__ == "__main__":
    main()
