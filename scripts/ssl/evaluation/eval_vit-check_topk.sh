#!/bin/bash

: '
Usage: sh scripts/robust/evaluation/fix_angle_eval.sh [epoch] [dir] [fixed_angle]

Arguments:
    epoch           Training epoch number to evaluate (default: 19)
    dir             Directory path (under $NAVSIM_EXP_ROOT) containing model checkpoints
                    (default: "train/ego_perturbation/rot_30-trans_0-p_0.5/fixbug-v1/baseline")  
'

# Default values
epoch=${1:-19}
dir=${2:-"training/ssl/ori/lr_baseline"}
num_top_k=${3:-64}
inference_model="teacher"


# Format epoch with leading zero
padded_epoch=$(printf "%02d" $epoch)

# Calculate step from epoch (1330 steps per epoch)
step=$((($epoch + 1) * 1330))

metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"

# Set experiment name based on inference model
if [ "$inference_model" = "teacher" ]; then
    experiment_name="${dir}/test-${padded_epoch}ep"
else
    experiment_name="${dir}/test-${padded_epoch}ep-${inference_model}"
fi

experiment_name="${experiment_name}-check_k_${num_top_k}"

command_string="TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_ssl.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=hydra_img_vit_ssl \
    dataloader.params.batch_size=8 \
    worker.threads_per_node=128 \
    agent.checkpoint_path='${NAVSIM_EXP_ROOT}/${dir}/epoch\=${padded_epoch}-step\=${step}.ckpt' \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=${inference_model} \
    agent.config.lab.check_top_k_traj=true \
    agent.config.lab.num_top_k=$num_top_k \
    agent.config.lab.test_full_vocab_pdm_score_path='${NAVSIM_TRAJPDM_ROOT}/ori/vocab_score_full_8192_navtest/navtest.pkl' \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path} \
    split=test \
    scene_filter=navtest
"

echo "--- COMMAND ---"
echo $command_string
echo -e "\n\n"

eval $command_string
