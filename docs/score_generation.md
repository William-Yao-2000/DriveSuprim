# Trajectory Score Generation

For each scenario in NAVSIM, we need to generate the rule-based metric score for the trajectories in the pre-defined trajectory vocabulary, which is used in model training. The full steps are as follows.

For debugging, you can implement the steps on the `navtrain_debug` split.

> Score Generation is time-consuming. You can also ignore the steps below and directly download our [generated score data](https://huggingface.co/alkaid-2000/DriveSuprim/blob/main/traj_pdm_v2/ori/vocab_score_8192_navtrain_final/navtrain.pkl).

### 1. Dataset Metric Caching

```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/ori \
    --config-name default_metric_caching
```

<details>
<summary>Debug split</summary>

```bash
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_debug \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/ori_debug \
    --config-name default_metric_caching
```
</details>

### 2. Generate PDM Score for the Full Vocabulary

In this step, we generate the PDM score for the pre-defined trajectory vocabulary on `navtrain` split.

This process can be time consuming. You can split the `navtrain` set into multiple subsets (e.g., `navtrain_sub1`, `navtrain_sub2`, ...) and parallelly run the script on several cpu machines.

```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score.py train_test_split=navtrain \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=navtrain \
    experiment_name=full_vocab_pdm_scoring/ori/navtrain \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/ori
```

<details>
<summary>Debug split</summary>

```bash
export PROGRESS_MODE=gen_gt; \
python navsim/agents/tools/gen_vocab_full_score.py train_test_split=navtrain_debug \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=navtrain_debug \
    experiment_name=full_vocab_pdm_scoring/ori/navtrain_debug \
    worker.threads_per_node=128 \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/ori_debug
```
</details>
