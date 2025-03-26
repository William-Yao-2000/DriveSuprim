#!/bin/bash

: '
Usage: sh scripts/robust/evaluation/fix_angle_eval.sh [epoch] [dir] [fixed_angle]

Arguments:
    epoch
    dir
    agent
'

# Default values
epoch=$1
dir=$2
agent=$3

# Format epoch with leading zero
padded_epoch=$(printf "%02d" $epoch)

metric_cache_path="/home/shiyil/work/zxli/navsim_workspace/exp2/navtest_metric_cache"

# Set experiment name based on inference model
experiment_name="${dir}/test-${padded_epoch}ep"

# rename ckpts
cd ${NAVSIM_EXP_ROOT}/${dir}

for file in epoch=*-step=*.ckpt; do
    epoch=$(echo $file | sed -n 's/.*epoch=\([0-9][0-9]\).*/\1/p')
    new_filename="epoch${epoch}.ckpt"
    mv "$file" "$new_filename"
done

cd ${NAVSIM_DEVKIT_ROOT}

command_string="TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu.py \
    +use_pdm_closed=false \
    agent=$agent \
    dataloader.params.batch_size=16 \
    worker.threads_per_node=128 \
    agent.checkpoint_path='${NAVSIM_EXP_ROOT}/${dir}/epoch${padded_epoch}.ckpt' \
    agent.pdm_gt_path=null \
    trainer.params.precision=32 \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path} \
    train_test_split=navtest
"

echo "--- COMMAND ---"
echo $command_string
echo -e "\n\n"

eval $command_string
