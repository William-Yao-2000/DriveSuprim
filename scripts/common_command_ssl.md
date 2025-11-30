## 1. testing
### 0-1. metric caching (ori-two_stage)
```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtest \
    worker.threads_per_node=192 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/test/ori \
    --config-name default_metric_caching
```

### 0-2. metric caching (warmup_two_stage)
```bash
python navsim/planning/script/run_metric_caching.py train_test_split=warmup_two_stage \
    worker.threads_per_node=192 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/warmup_two_stage \
    synthetic_sensor_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs \
    synthetic_scenes_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles \
    --config-name default_metric_caching
```

### 0-3. metric caching (navhard_two_stage)
```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navhard_two_stage \
    worker.threads_per_node=192 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navhard_two_stage \
    --config-name default_metric_caching
```


### 1. ori
```bash
TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu.py \
    +use_pdm_closed=false \
    agent=hydra_img_vov \
    dataloader.params.batch_size=8 \
    worker.threads_per_node=128 \
    agent.checkpoint_path=${NAVSIM_EXP_ROOT}/ckpt/hydra_img_vov.ckpt \
    experiment_name=ori_test/hydra_img_vov \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache/test/ori \
    split=test \
    scene_filter=navtest
```


debug
```bash
TORCH_NCCL_ENABLE_MONITORING=0 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_ssl.py \
    +debug=true \
    +use_pdm_closed=false \
    agent=hydra_img_vit_ssl \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    agent.checkpoint_path="${NAVSIM_EXP_ROOT}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/stage_layers_3-topks_256/epoch\=05-step\=7980.ckpt" \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache/test/ori \
    train_test_split=navtest
```

### 2. warmup_two_stage
debug (v2, not debug mode)
```bash
TORCH_NCCL_ENABLE_MONITORING=0 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_ssl.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=hydra_img_vit_ssl_v1 \
    dataloader.params.batch_size=2 \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    agent.checkpoint_path="${NAVSIM_EXP_ROOT}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage_v1/stage_layers_3-topks_256/epoch\=05-step\=7980.ckpt" \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache/warmup_two_stage \
    train_test_split=warmup_two_stage
```


debug (v1)
```bash
TORCH_NCCL_ENABLE_MONITORING=0 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_ssl_v1.py \
    +debug=true \
    +use_pdm_closed=false \
    agent=hydra_img_vit_ssl_v1 \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    agent.checkpoint_path="${NAVSIM_EXP_ROOT}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage_v1/stage_layers_3-topks_256/epoch\=05-step\=7980.ckpt" \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache/warmup_two_stage \
    train_test_split=warmup_two_stage
```


debug (v1, not debug mode)
```bash
TORCH_NCCL_ENABLE_MONITORING=0 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_gpu_ssl_v1.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=hydra_img_vit_ssl_v1 \
    dataloader.params.batch_size=2 \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    agent.checkpoint_path="${NAVSIM_EXP_ROOT}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage_v1/stage_layers_3-topks_256/epoch\=05-step\=7980.ckpt" \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache/warmup_two_stage \
    train_test_split=warmup_two_stage
```


### 3. submission
#### 3-1. warmup_two_stage

debug
```bash
TORCH_NCCL_ENABLE_MONITORING=0 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle_warmup.py \
    train_test_split=warmup_two_stage \
    agent=hydra_img_vit_ssl \
    dataloader.params.batch_size=8 \
    agent.checkpoint_path="${NAVSIM_EXP_ROOT}/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/stage_layers_3-topks_256/epoch\=05-step\=7980.ckpt" \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=teacher \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    agent.config.lab.use_first_stage_traj_in_infer=true \
    experiment_name=debug_submission \
    train_test_split=warmup_two_stage \
    team_name=ntestv_1 \
    authors=why \
    email=whyao23@m.fudan.edu.cn \
    institution=fdu \
    country=chn \
    synthetic_sensor_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs \
    synthetic_scenes_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles
```


## 2. training
### 0. generate offline augmentation file
```bash
python navsim/agents/tools/gen_offline_training_aug_file.py --rot=45 --trans=0 --va=0 --percentage=1.0 --seed=2024
```
参数：
    rot: 角度制
    trans: （暂时忽略）
    vel: 百分比，如 0.3 表示速度在原来的 0.7 倍和 1.3 倍之间扰动
    acc: 百分比
    percentage: 增强数据所占全部数据的比例

ensemble offline files:
```bash
python navsim/agents/tools/gen_offline_training_aug_file_ensemble_seed.py --rot=45 --trans=0 --va=0 --percentage=1.0
```

### 1. metric caching (augmentation)

ori
```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain \
    worker.threads_per_node=192 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/ori \
    --config-name default_metric_caching
```

augmentation
```bash
python navsim/planning/script/run_metric_caching_aug_train.py train_test_split=navtrain \
    +debug=false \
    worker.threads_per_node=192 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug/rot_45-trans_0-va_0-p_1.0-seed_2026 \
    aug_train.rotation=45 \
    aug_train.va=0 \
    offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_45-trans_0-va_0-p_1.0-seed_2026.json \
    --config-name metric_caching_aug_train
```

debug
```bash
python navsim/planning/script/run_metric_caching_aug_train.py train_test_split=navtrain_debug \
    +debug=true \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    metric_cache_path=$NAVSIM_EXP_ROOT/debug \
    aug_train.rotation=0 \
    aug_train.va=0.3 \
    offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/va_0.3-p_0.5.json\
    --config-name metric_caching_aug_train
```


### 2. gen full vocab pdm score

bash command (subset score generation)
```bash
bash scripts/ssl/gen_full_score_aug/gen_training_full_score_aug_subset-seeds.sh navtrain_ngc_sub12 2026
```



debug
```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score_aug_train.py train_test_split=navtrain_debug \
    +debug=true \
    +vocab_size=8192 \
    +scene_filter_name=navtrain_debug \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/debug \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    aug_train.rotation=30 \
    aug_train.seed=2026 \
    experiment_name=debug_full_vocab_pdm_scoring_aug \
    +save_name=navtrain_debug
```

debug (not debug mode)
```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score_aug_train.py train_test_split=navtrain_debug \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=navtrain_debug \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/debug/train/random_aug/rot_30-trans_0-va_0-p_0.5-seed_2024 \
    worker.threads_per_node=128 \
    aug_train.rotation=30 \
    aug_train.seed=2024 \
    experiment_name=debug
```

ori
```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score.py train_test_split=navtrain_ngc_sub1 \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=navtrain_ngc_sub1 \
    experiment_name=full_vocab_pdm_scoring_aug/ori/navtrain_ngc_sub1 \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/ori
```

ori debug
```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score.py train_test_split=navtrain_debug \
    +debug=true \
    +vocab_size=8192 \
    +scene_filter_name=navtrain_debug \
    experiment_name=debug_full_vocab_pdm_scoring_aug \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/debug/train/ori
```

ori debug (not debug mode)
```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score.py train_test_split=navtrain_debug \
    +debug=false \
    +scene_filter_name=navtrain_debug \
    +vocab_size=8192 \
    experiment_name=debug \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/debug/train/ori
```

**ensemble**
```bash
python navsim/agents/tools/get_final_full_vocab_score.py --rot=45 --trans=0 --va=0 --percentage=1.0 --seed=2026

python navsim/agents/tools/get_final_full_vocab_score_ensemble_seeds.py --rot=45 --trans=0 --va=0 --percentage=1.0
```

**split emsembles**
```bash
python navsim/agents/tools/split_final_ensemble_pickle.py --rot=45 --trans=0 --va=0 --percentage=1.0
```


ori-test
```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score.py train_test_split=navtest \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=navtest \
    experiment_name=full_vocab_pdm_scoring_aug/ori/navtest \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/test/ori
```


### 3. get close pdm traj from metric caching

由于 cache（第一步）的时候没有分子集，而是全部一起 cache 的，因此这里也可以直接整个集合一起做 get close pdm traj，不用分集合

```bash
python navsim/agents/scripts/get_pdm_closed_traj_from_metric_cache.py split=trainval scene_filter=navtrain \
    +debug=false \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug/rot_30-trans_0-p_0.5 \
    worker.threads_per_node=128 \
    experiment_name=debug \
    +save_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug/rot_30-trans_0-p_0.5/pdm_closed_traj.pkl
```

debug
```bash
python navsim/agents/scripts/get_pdm_closed_traj_from_metric_cache.py split=trainval scene_filter=navtrain_debug \
    +debug=true \
    metric_cache_path=$NAVSIM_EXP_ROOT/debug_metric_cache \
    experiment_name=debug \
    +save_path=$NAVSIM_EXP_ROOT/debug/pdm_closed_traj.pkl
```

debug (not debug mode)
```bash
python navsim/agents/scripts/get_pdm_closed_traj_from_metric_cache.py split=trainval scene_filter=navtrain_debug \
    +debug=false \
    metric_cache_path=$NAVSIM_EXP_ROOT/debug_metric_cache \
    worker.threads_per_node=128 \
    experiment_name=debug \
    +save_path=$NAVSIM_EXP_ROOT/debug_metric_cache/pdm_closed_traj.pkl
```


### 4. training

bash command
```bash
sh scripts/robust/training/node_aug.sh
```


debug (只是为了测试，有可能参数之间并没有相符/对齐)

teacher + student, ori input + rotate input (3-ensemble)
其实跟上面没什么变化，只是代码变了
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=true \
    agent=drivesuprim_agent_vit \
    experiment_name=debug \
    split=trainval \
    train_test_split=navtrain_debug \
    \~trainer.params.strategy \
    trainer.params.limit_train_batches=0.08 \
    trainer.params.limit_val_batches=0.20 \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    agent.config.ego_perturb.n_student_rotation_ensemble=3 \
    agent.config.ego_perturb.offline_aug_angle_boundary=30 \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_30-p_0.5-ensemble.json \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=1 \
    agent.config.refinement.stage_layers=3 \
    agent.config.refinement.topks=256 \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_debug/navtrain_debug.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_30-p_0.5-ensemble_debug/split_pickles \
    cache_path=null
```

debug (not debug mode)

```bash
CUDA_VISIBLE_DEVICES=0 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=false \
    agent=hydra_img_vit_ssl \
    experiment_name=training/debug \
    split=trainval \
    scene_filter=navtrain_debug \
    dataloader.params.batch_size=4 \
    \~trainer.params.strategy \
    trainer.params.limit_train_batches=0.04 \
    trainer.params.limit_val_batches=0.20 \
    trainer.params.max_epochs=20 \
    agent.config.ckpt_path=training/debug \
    agent.config.ego_perturb.mode=load_from_offline \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_30-trans_0-va_0-p_0.5-ensemble.json \
    agent.config.ego_perturb.rotation.enable=true \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=30 \
    agent.config.student_rotation_ensemble=3 \
    agent.config.soft_label_diff_thresh=1.0 \
    agent.config.refinement.use_2_stage=true \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_debug/navtrain_debug.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_30-trans_0-va_0-p_0.5-ensemble_debug/split_pickles \
    cache_path=null
```


## 3. visualization


single_stage
```bash
python ${NAVSIM_DEVKIT_ROOT}/navsim/visualization/navtest-single_stage.py \
    +debug=false \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=64 \
    experiment_name=debug \
    train_test_split=navtest_vis \
    +start_idx=0 \
    +end_idx=200
```


multi_stage
```bash
python ${NAVSIM_DEVKIT_ROOT}/navsim/visualization/navtest-multi_stage.py \
    +debug=false \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=64 \
    experiment_name=debug \
    train_test_split=navtest_vis \
    +start_idx=0 \
    +end_idx=200
```


debug
```bash
python ${NAVSIM_DEVKIT_ROOT}/navsim/visualization/navtest-multi_stage.py \
    +debug=true \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    experiment_name=debug \
    train_test_split=navtest_vis \
    +start_idx=0 \
    +end_idx=1000
```


split navtest angle:
```bash
python ${NAVSIM_DEVKIT_ROOT}/temp-test-code/split_navtest_angle.py \
    +debug=false \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=192 \
    experiment_name=debug \
    train_test_split=navtrain
```


compare pred score with gt score:
debug
```bash
python ${NAVSIM_DEVKIT_ROOT}/navsim/visualization/score_comparison.py \
    +debug=true \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    experiment_name=debug \
    train_test_split=navtest_vis_1 \
    +start_idx=0 \
    +end_idx=1000
```