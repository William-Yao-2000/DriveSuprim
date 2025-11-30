# Training and Evaluation

## Training

### 1. Download pre-trained weight of backbones

TODO

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


## Evaluation

### 1. Metric caching for test set

```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtest \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/test/ori \
    --config-name default_metric_caching
```

### 2. Run evaluation script

```bash
# ResNet34
bash scripts/drivesuprim/evaluation/eval.sh \
    9 training/drivesuprim_agent_r34/rot_30-p_0.5/stage_layers_3-topks_256 \
    1 3 256 drivesuprim_agent_r34 teacher

# V2-99
bash scripts/drivesuprim/evaluation/eval.sh \
    9 training/drivesuprim_agent_vov/rot_30-p_0.5/stage_layers_3-topks_256 \
    1 3 256 drivesuprim_agent_vov teacher

# ViT-Large
bash scripts/drivesuprim/evaluation/eval.sh \
    5 training/drivesuprim_agent_vit/rot_30-p_0.5/stage_layers_3-topks_256 \
    1 3 256 drivesuprim_agent_vit teacher
```
