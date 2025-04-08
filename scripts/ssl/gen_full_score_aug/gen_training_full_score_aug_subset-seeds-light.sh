#!/bin/bash
scene_filter=$1
seed=$2

rot=30
trans=0
va=0
percentage=0.5


export PROGRESS_MODE='gen_gt'  # IMPORTANT!!!



if [ -z "$scene_filter" ]; then
    echo "Wrong! The proper command is: \nsh scripts/ssl/gen_full_score_aug/gen_training_full_score_aug_subset-seeds.sh [scene_filter_name]"
    exit 1
fi

if [ $rot -eq 0 ] && [ $trans -eq 0 ] && [ $(echo "$va == 0" | bc -l) -eq 1 ]; then
    echo "Error: At least one of rot, trans, or va must be non-zero"
    exit 1
fi


aug_str="rot_${rot}-trans_${trans}-va_${va}-p_${percentage}-seed_${seed}"

command="python navsim/agents/tools/gen_vocab_full_score_aug_train.py train_test_split=$scene_filter \
    +debug=false \
    +vocab_size=8192 \
    +scene_filter_name=$scene_filter \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/train/random_aug/${aug_str} \
    worker.threads_per_node=64 \
    aug_train.rotation=$rot \
    aug_train.translation=$trans \
    aug_train.va=$va \
    aug_train.percentage=$percentage \
    aug_train.seed=$seed \
    experiment_name=full_vocab_pdm_scoring_aug/${aug_str}/$scene_filter"

echo "--- COMMAND ---"
echo $command
echo -e "\n\n"

eval $command
