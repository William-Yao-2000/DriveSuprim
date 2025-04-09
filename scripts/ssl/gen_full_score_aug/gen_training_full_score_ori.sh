#!/bin/bash
scene_filter=$1

export PROGRESS_MODE='gen_gt'  # IMPORTANT!!!


if [ -z "$scene_filter" ]; then
    echo "Wrong! The proper command is: \nsh scripts/ssl/gen_full_score_aug/gen_training_full_score_aug_subset-seeds.sh [scene_filter_name]"
    exit 1
fi

command="python navsim/agents/tools/gen_vocab_full_score.py train_test_split=$scene_filter \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=$scene_filter \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/ori \
    worker.threads_per_node=128 \
    experiment_name=full_vocab_pdm_scoring_aug/ori/$scene_filter"

echo "--- COMMAND ---"
echo $command
echo -e "\n\n"

eval $command
