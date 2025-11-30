#!/bin/bash

# Default values
epoch=$1
dir=${2:-"training/ssl/ori/lr_baseline"}
num_refinement_stage=$3
stage_layers=$4
topks=$5
agent=${6:-"drivesuprim_agent_r34"}
inference_model=${7:-"teacher"}

# Format epoch with leading zero
padded_epoch=$(printf "%02d" $epoch)

# Calculate step from epoch (1330 steps per epoch)
step=$((($epoch + 1) * 1330))

metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"

# Set experiment name based on inference model
if [ "$inference_model" = "teacher" ]; then
    experiment_name="${dir}/test-${padded_epoch}ep-one_stage"
else
    experiment_name="${dir}/test-${padded_epoch}ep-${inference_model}-one_stage"
fi

command_string="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=$agent \
    train_test_split=navtest \
    dataloader.params.batch_size=8 \
    worker.threads_per_node=128 \
    agent.checkpoint_path='${NAVSIM_EXP_ROOT}/${dir}/epoch=${padded_epoch}-step=${step}.ckpt' \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=${inference_model} \
    agent.config.inference.save_pickle=false \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path}
"

echo "--- COMMAND ---"
echo $command_string
echo -e "\n\n"

torchrun --nproc_per_node=8 --master_port=29500 $command_string


: '
usage:
bash scripts/drivesuprim/evaluation/eval.sh \
    8 training/drivesuprim_agent_r34/rot_30-p_0.5/stage_layers_3-topks_256 \
    1 3 256 drivesuprim_agent_r34 teacher
'