# Augmentation Data Generation

For each scenario in NAVSIM, we need to generate the rotation-based augmentation data.\
Specifically, we specify several rotation angles for each scenario, revise the ego pose, and generate the pdm score for each augmented scenario. The main procedure is the same as the score generation.

Augmentation Data Generation is time-consuming. You can also directly download our generated augmentation data.

(For debugging, you can implement the steps on the `navtrain_debug` split.)


### 1. Generate offline augmentation file

Each augmentation file specifies a random seed, and assigns a rotation angle to each scenario.\
We ensemble multiple augmentation files to form larger augmented dataset, which means that each scenario is related to multiple rotation angles.

#### 1.1 Generate augmentation file

You can change the rotation boundary, augmentation percentage, and random seed to generate different augmentation files.

```bash
python navsim/agents/tools/gen_offline_training_aug_file.py --rot=30 --percentage=0.5 --seed=2024
```

#### 1.2 Offline file ensemble

This ensembled file is used in model training. Here we ensemble files with random seed from 2024 to 2026.

```bash
python navsim/agents/tools/gen_offline_training_aug_file_ensemble_seed.py --rot=30 --percentage=0.5 --begin_seed=2024 --end_seed=2026
```


### 2. Augmented Dataset Metric Cache

Please make sure that the rotation angle, percentage, and random seed in **offline_aug_file** is the same as the one in **metric_cache_path**.

```bash
python navsim/planning/script/run_metric_caching_aug_train.py train_test_split=navtrain \
    +debug=false \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug/rot_30-p_0.5-seed_2024 \
    offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_30-p_0.5-seed_2024.json \
    --config-name metric_caching_aug_train
```

<details>
<summary>Debug split</summary>

```bash
python navsim/planning/script/run_metric_caching_aug_train.py train_test_split=navtrain_debug \
    +debug=false \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug_debug/rot_30-p_0.5-seed_2024 \
    offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_30-p_0.5-seed_2024.json \
    --config-name metric_caching_aug_train
```
</details>

### 3. Generate PDM Score for the Full Vocabulary with Augmented Data

#### 3.1 Generate PDM Score

This process can be time consuming. You can split the `navtrain` set into multiple subsets (e.g., `navtrain_sub1`, `navtrain_sub2`, ...) and parallelly run the script on several cpu machines.

```bash
bash scripts/drivesuprim/gen_full_score_aug/gen_training_full_score_aug.sh 30 0.5 2024
```

<details>
<summary>Debug split</summary>

```bash
export PROGRESS_MODE=gen_gt; \
export _rot=30; \
export _percentage=0.5; \
export _seed=2026; \
python navsim/agents/tools/gen_vocab_full_score_aug_train.py train_test_split=navtrain_debug \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=navtrain_debug \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug_debug/rot_${_rot}-p_${_percentage}-seed_${_seed} \
    worker.threads_per_node=128 \
    aug_train.rotation=${_rot} \
    aug_train.percentage=${_percentage} \
    aug_train.seed=${_seed} \
    experiment_name=full_vocab_pdm_scoring_aug/rot_${_rot}-p_${_percentage}-seed_${_seed}/navtrain_debug
```

</details>


#### 3.2 Ensemble and split into pickles

##### 3.2.1 Ensemble

Ensemble the PDM scores across multiple seeds.

```bash
python navsim/agents/tools/get_final_full_vocab_score_ensemble_seeds.py --rot=30 --percentage=0.5 --begin_seed=2024 --end_seed=2026
```

<details>
<summary>Debug split</summary>

```bash
python navsim/agents/tools/get_final_full_vocab_score_ensemble_seeds.py --rot=30 --percentage=0.5 --begin_seed=2024 --end_seed=2026 --debug_split
```

</details>

##### 3.2.2 Split emsembled pkl file

Split the ensembled file into multiple pickle files named by the scenario token, which are used in training.

```bash
python navsim/agents/tools/split_final_ensemble_pickle.py --rot=30 --percentage=0.5
```

<details>
<summary>Debug split</summary>

```bash
python navsim/agents/tools/split_final_ensemble_pickle.py --rot=30 --percentage=0.5 --debug_split
```

</details>