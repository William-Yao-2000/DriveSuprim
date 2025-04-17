## 1. testing
### 0. metric caching (ori / augmentation)
```bash
python navsim/planning/script/run_metric_caching.py split=test scene_filter=navtest \
    worker.threads_per_node=128 \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache/test/ori \
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
    agent.config.inference.model=student \
    agent.config.refinement.use_2_stage=true \
    agent.config.lab.test_full_vocab_pdm_score_path='/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/hydramdp_cvpr24/dataset/traj_pdm/ori/vocab_score_8192_navtest_final/navtest.pkl' \
    agent.checkpoint_path='/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp/training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/ensemble_3-lr3x/epoch\=01-step\=2660.ckpt' \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=/DATA3/yaowenhao/proj/auto_drive/navsim_workspace/exp/metric_cache/test/ori \
    split=test \
    scene_filter=navtest
```


### 2. camera_shutdown
```bash
TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_robust.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=hydra_img_vov_robust \
    dataloader.params.batch_size=8 \
    worker.threads_per_node=128 \
    agent.checkpoint_path=${NAVSIM_EXP_ROOT}/ckpt/hydra_img_vov.ckpt \
    agent.config.camera_shutdown=True \
    agent.config.camera_shutdown_mode=1 \
    experiment_name=test/camera_shutdown/mode_1/p_1.0 \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/navtest_metric_cache \
    split=test \
    scene_filter=navtest
```

debug
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_robust.py \
    +debug=true \
    +use_pdm_closed=false \
    agent=hydra_img_vov_robust \
    dataloader.params.batch_size=8 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    agent.checkpoint_path=${NAVSIM_EXP_ROOT}/ckpt/hydra_img_vov.ckpt \
    agent.config.camera_shutdown=True \
    agent.config.camera_shutdown_mode=1 \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/navtest_metric_cache \
    split=test \
    scene_filter=navtest_debug
```

### 3. camera noise
debug
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_robust.py \
    +debug=true \
    +use_pdm_closed=false \
    agent=hydra_img_vov_robust \
    dataloader.params.batch_size=8 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    agent.checkpoint_path=${NAVSIM_EXP_ROOT}/ckpt/hydra_img_vov.ckpt \
    agent.config.camera_noise=True \
    agent.config.camera_noise_percentage=0.3 \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/navtest_metric_cache \
    split=test \
    scene_filter=navtest_debug
```


### 4. ego_rotation
#### crop from panoramic img
```bash
TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_robust.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=hydra_img_vov_robust \
    dataloader.params.batch_size=8 \
    worker.threads_per_node=128 \
    agent.checkpoint_path="${NAVSIM_EXP_ROOT}/train/ego_perturbation/rot_30-trans_0-p_0.5/fixbug-v1/traj_smooth/epoch\=05-step\=7980.ckpt" \
    agent.config.ego_perturb.rotation.enable=true \
    agent.config.ego_perturb.rotation.fixed_angle=0 \
    agent.config.ego_perturb.rotation.crop_from_panoramic=true \
    agent.config.training=false \
    experiment_name=train/ego_perturbation/rot_30-trans_0-p_0.5/fixbug-v1/traj_smooth/test-05ep \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache/test/ori \
    split=test \
    scene_filter=navtest
```


debug （参数可能对不上，只是为了演示）
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_robust.py \
    +debug=true \
    +use_pdm_closed=false \
    agent=hydra_img_vov_robust \
    dataloader.params.batch_size=8 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    agent.checkpoint_path=${NAVSIM_EXP_ROOT}/ckpt/hydra_img_vov.ckpt \
    agent.config.ego_perturb.mode=load_from_offline \
    agent.config.ego_perturb.offline_aug_file=${NAVSIM_EXP_ROOT}/offline_files/testing_aug_files/rot_15-trans_0-va_0.3-wp_0.2-p_1.0.json \
    agent.config.ego_perturb.rotation.enable=true \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=15 \
    agent.config.ego_perturb.rotation.crop_from_panoramic=true \
    agent.config.ego_perturb.va.enable=true \
    agent.config.ego_perturb.va.offline_aug_boundary=0.3 \
    agent.config.camera_problem.weather_enable=true \
    agent.config.camera_problem.weather_aug_mode=load_from_offline \
    agent.config.training=false \
    experiment_name=debug \
    +cache_path=null \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache/test/fixed_aug/rot_15-trans_0-p_1.0 \
    split=test \
    scene_filter=navtest_debug
```

### 5. testing on augmented dataset
#### 1-1. generate offline files
```bash
python navsim/agents/scripts/gen_offline_testing_aug_file.py --rot=0 --va=0.3
```


#### 2. metric caching
虽然运行的是 run_metric_caching_aug_train.py，但实际上是在测试集上面 cache 的
```bash
python navsim/planning/script/run_metric_caching_aug_train.py split=test scene_filter=navtest \
    +debug=false \
    worker.threads_per_node=192 \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache/test/random_aug/M_2 \
    aug_train.rotation=45 \
    aug_train.va=0.5 \
    offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/testing_aug_files/rot_45-trans_0-va_0.5-wp_0.3-p_1.0.json \
    --config-name metric_caching_aug_train
```


debug
```bash
python navsim/planning/script/run_metric_caching_aug_train.py split=test scene_filter=navtest \
    +debug=true \
    cache.cache_path=$NAVSIM_EXP_ROOT/debug \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    aug_train.rotation=15 \
    aug_train.va=0 \
    offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/testing_aug_files/rot_15-trans_0-p_1.0.json \
    --config-name metric_caching_aug_train
```



### --2. metric caching (augmentation)
1. rotation
```bash
python navsim/planning/script/run_metric_caching_aug_test.py split=test scene_filter=navtest \
    +debug=false \
    worker.threads_per_node=128 \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache/test/fixed_aug/rot_-30-trans_0-p_1.0 \
    augmentation.rotation=-30 \
    --config-name metric_caching_aug
```

debug
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python navsim/planning/script/run_metric_caching_aug.py split=test scene_filter=navtest_debug \
    +debug=true \
    cache.cache_path=$NAVSIM_EXP_ROOT/debug \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    augmentation.rotation=20.0 \
    --config-name metric_caching_aug
```

## 2. training
### 0. generate offline augmentation file
```bash
python navsim/agents/scripts/gen_offline_training_aug_file.py --rot=30 --trans=0 --va=0 --percentage=0.5 --seed=2025
```
参数：
    rot: 角度制
    trans: （暂时忽略）
    vel: 百分比，如 0.3 表示速度在原来的 0.7 倍和 1.3 倍之间扰动
    acc: 百分比
    percentage: 增强数据所占全部数据的比例

ensemble offline files:
```bash
python navsim/agents/scripts/gen_offline_training_aug_file_ensemble_seed.py --rot=30 --trans=0 --va=0 --percentage=0.5
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
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug/rot_30-trans_0-va_0-p_0.5-seed_2025 \
    aug_train.rotation=30 \
    aug_train.va=0 \
    offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_30-trans_0-va_0-p_0.5-seed_2025.json \
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
bash scripts/ssl/gen_full_score_aug/gen_training_full_score_aug_subset-seeds.sh navtrain_ngc_sub1 2024
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
python navsim/agents/tools/get_final_full_vocab_score.py --rot=30 --trans=0 --va=0 --percentage=0.5 --seed=2024

python navsim/agents/tools/get_final_full_vocab_score_ensemble_seeds.py --rot=30 --trans=0 --va=0 --percentage=0.5
```

**split emsembles**
```bash
python navsim/agents/tools/split_final_ensemble_pickle.py --rot=30 --trans=0 --va=0 --percentage=0.5
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

only student, only ori input
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=true \
    agent=hydra_img_vit_ssl \
    experiment_name=debug \
    split=trainval \
    scene_filter=navtrain_debug \
    \~trainer.params.strategy \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    agent.config.ego_perturb.mode=load_from_offline \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_0-trans_0-p_0.0.json \
    agent.config.ego_perturb.rotation.enable=false \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=0 \
    agent.config.only_ori_input=true \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_debug/navtrain_debug.pkl \
    cache_path=null
```

only student, ori input + rotate input (3-ensemble)
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=true \
    agent=hydra_img_vit_ssl \
    experiment_name=debug \
    split=trainval \
    scene_filter=navtrain_debug \
    \~trainer.params.strategy \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    agent.config.ego_perturb.mode=load_from_offline \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_30-trans_0-va_0-p_0.5-ensemble.json \
    agent.config.ego_perturb.rotation.enable=true \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=30 \
    agent.config.student_rotation_ensemble=3 \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_debug/navtrain_debug.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_30-trans_0-va_0-p_0.5-ensemble_debug/split_pickles \
    cache_path=null
```

teacher + student, ori input + rotate input (3-ensemble)
其实跟上面没什么变化，只是代码变了
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=true \
    agent=hydra_img_vit_ssl \
    experiment_name=debug \
    split=trainval \
    train_test_split=navtrain_debug \
    \~trainer.params.strategy \
    trainer.params.limit_train_batches=0.08 \
    trainer.params.limit_val_batches=0.20 \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=0 \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=null \
    agent.config.ego_perturb.mode=load_from_offline \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_30-trans_0-va_0-p_0.5-ensemble.json \
    agent.config.ego_perturb.rotation.enable=true \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=30 \
    agent.config.student_rotation_ensemble=3 \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=2 \
    agent.config.refinement.stage_layers="3+2" \
    agent.config.refinement.topks="64+16" \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_debug/navtrain_debug.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_30-trans_0-va_0.0-p_0.5-ensemble_debug/split_pickles \
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

```bash
python ${NAVSIM_DEVKIT_ROOT}/navsim/visualization/navtest_robust_aug.py \
    +debug=true \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=64 \
    experiment_name=debug \
    split=test \
    scene_filter=navtest_aug_select \
    +offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/testing_aug_files/rot_45-trans_0-va_0.5-wp_0.3-p_1.0.json \
    +start_idx=0 \
    +end_idx=200
```


debug
```bash
python ${NAVSIM_DEVKIT_ROOT}/navsim/visualization/navtest_robust.py \
    +debug=true \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=0 \
    worker.debug_mode=true \
    experiment_name=debug \
    split=test \
    scene_filter=navtest_aug_select \
    +offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/testing_aug_files/rot_45-trans_0-va_0.5-wp_0.3-p_1.0.json \
    +start_idx=0 \
    +end_idx=1000
```