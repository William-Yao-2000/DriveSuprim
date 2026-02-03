# Training and Evaluation

## Training

Before training, please make sure you have prepared all the required data, including the NAVSIM dataset, the offline augmentation file, and the PDM Score data. Your directory should be assigned as follows:

```angular2html
~/drivesuprim_workspace
├── DriveSuprim
├── dataset
│   ├── maps
│   ├── navsim_logs
│   ├── sensor_blobs
│   └── traj_pdm_v2
│       ├── ori
│       │   └── vocab_score_8192_navtrain_final
│       │       └── navtrain.pkl
│       └── random_aug
│           └── rot_30-p_0.5-ensemble
│               └── vocab_score_8192_navtrain_final
│                   └── split_pickles
└── exp_v2
    ├── offline_files
    │   └── training_ego_aug_files
    │       └── rot_30-p_0.5-ensemble.json
    └── models
```

### 1. Download pre-trained weight of backbones

Before running the training script, you need to download the pre-trained weight of backbones, including V2-99 and ViT-Large.

- V2-99: [dd3d_det_final.pth](https://huggingface.co/alkaid-2000/DriveSuprim/resolve/main/pretrained_backbones/dd3d_det_final.pth)
- ViT-Large: [da_vitl16.pth](https://huggingface.co/alkaid-2000/DriveSuprim/resolve/main/pretrained_backbones/da_vitl16.pth)

Then place them in the `$NAVSIM_EXP_ROOT/models` folder.

```angular2html
~/drivesuprim_workspace
├── DriveSuprim
├── dataset
└── exp_v2
    └── models
         ├── da_vitl16.pth
         └── dd3d_det_final.pth
```

### 2. Run training script
```bash
# ResNet34
bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh \
  drivesuprim_agent_r34 1 3 256

# V2-99
bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh \
  drivesuprim_agent_vov 1 3 256

# ViT-Large
bash scripts/drivesuprim/training/rot_30-p_0.5/train.sh \
  drivesuprim_agent_vit 1 3 256
```

<details>
<summary>Debug split training</summary>

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
    trainer.params.limit_val_batches=0.50 \
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
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_30-p_0.5-ensemble_debug/vocab_score_8192_navtrain_final/split_pickles \
    cache_path=null
```
</details>


## Evaluation

### 1. Metric caching for test set

```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtest \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/test/ori \
    --config-name default_metric_caching
```

### 2. Run evaluation script

We provide two bash file for evaluation. You can choose one of them to run according to your need.
- `eval_epoch.sh`: evaluate the model on a specific epoch
- `eval_file.sh`: evaluate the model on a specific checkpoint file

#### 2.1 Evaluate on a specific epoch

After training for several epochs, you can evaluate the model on a specific epoch.

```bash
# ResNet34
bash scripts/drivesuprim/evaluation/eval_epoch.sh \
    9 training/drivesuprim_agent_r34/rot_30-p_0.5/stage_layers_3-topks_256 \
    1 3 256 drivesuprim_agent_r34 teacher

# V2-99
bash scripts/drivesuprim/evaluation/eval_epoch.sh \
    9 training/drivesuprim_agent_vov/rot_30-p_0.5/stage_layers_3-topks_256 \
    1 3 256 drivesuprim_agent_vov teacher

# ViT-Large
bash scripts/drivesuprim/evaluation/eval_epoch.sh \
    5 training/drivesuprim_agent_vit/rot_30-p_0.5/stage_layers_3-topks_256 \
    1 3 256 drivesuprim_agent_vit teacher
```


#### 2.2 Evaluate on a specific checkpoint file

You can download our model checkpoint from [here](https://huggingface.co/alkaid-2000/DriveSuprim/tree/main/model_ckpt), and then move to exp_v2 directory (shown below). For more details, please refer to the [README](../README.md) file.


```angular2html
~/drivesuprim_workspace
├── DriveSuprim
├── dataset
└── exp_v2
    └── model_ckpt
         ├── drivesuprim_r34.ckpt
         ├── drivesuprim_vit.ckpt
         └── drivesuprim_vov.ckpt
```


```bash
# ResNet34
bash scripts/drivesuprim/evaluation/eval_file.sh \
    model_ckpt drivesuprim_r34 \
    1 3 256 drivesuprim_agent_r34 teacher

# V2-99
bash scripts/drivesuprim/evaluation/eval_file.sh \
    model_ckpt drivesuprim_vov \
    1 3 256 drivesuprim_agent_vov teacher

# ViT-Large
bash scripts/drivesuprim/evaluation/eval_file.sh \
    model_ckpt drivesuprim_vit \
    1 3 256 drivesuprim_agent_vit teacher
```
