import logging

import hydra
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig

from navsim.planning.metric_caching.caching_aug_train import cache_data_aug_train
from navsim.planning.script.builders.worker_pool_builder import build_worker

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/metric_caching"
CONFIG_NAME = "default_metric_caching"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for metric caching.
    :param cfg: omegaconf dictionary
    """
    # Configure logger
    build_logger(cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.debug:
        import os; os.environ['ROBUST_CACHING_DEBUG'] = 'true'
        import pdb; pdb.set_trace()

    # Precompute and cache all features
    logger.info("Starting Metric Caching...")
    if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
        raise AssertionError("ray in distributed mode will not work with this job")
    cache_data_aug_train(cfg=cfg, worker=worker)


if __name__ == "__main__":
    main()
