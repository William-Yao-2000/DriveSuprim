#!/bin/bash

# Default values
epoch=${1:-19}
dir=${2:-"training/ssl/ori/lr_baseline"}
num_refinement_stage=1
stage_layers=3
topks=256

refinement_metrics=$3

inference_model="teacher"

# Format epoch with leading zero
padded_epoch=$(printf "%02d" $epoch)

# Calculate step from epoch (1330 steps per epoch)
step=$((($epoch + 1) * 1330))

metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"

# Set experiment name based on inference model
if [ "$inference_model" = "teacher" ]; then
    experiment_name="${dir}/test-${padded_epoch}ep-one_stage-refinement_$refinement_metrics"
else
    experiment_name="${dir}/test-${padded_epoch}ep-${inference_model}-one_stage-refinement_$refinement_metrics"
fi

command_string="TORCH_NCCL_ENABLE_MONITORING=0 \
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py \
    +debug=false \
    +use_pdm_closed=false \
    agent=hydra_img_vit_ssl \
    dataloader.params.batch_size=8 \
    worker.threads_per_node=128 \
    agent.checkpoint_path='${NAVSIM_EXP_ROOT}/${dir}/epoch\=${padded_epoch}-step\=${step}.ckpt' \
    agent.config.training=false \
    agent.config.only_ori_input=true \
    agent.config.inference.model=${inference_model} \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    agent.config.lab.use_first_stage_traj_in_infer=false \
    agent.config.lab.refinement_metrics=$refinement_metrics \
    experiment_name=${experiment_name} \
    +cache_path=null \
    metric_cache_path=${metric_cache_path} \
    train_test_split=navtest \
"

echo "--- COMMAND ---"
echo $command_string
echo -e "\n\n"

eval $command_string


: '
usage:
bash scripts/ssl/evaluation/lab/eval_vit-multi_stage-partial_refinement_metrics.sh \
    4 training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs/stage_layers_3-topks_256-refinement_metrics_dac_ep_lk_pdms \
    dac_ep_lk_pdms
'