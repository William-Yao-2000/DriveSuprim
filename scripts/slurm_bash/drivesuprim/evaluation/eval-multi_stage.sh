#!/bin/bash
epoch=$1
dir=$2
agent=$3
partition=$4

# Parse additional arguments
# TODO: use dir to get these parameters
for arg in "${@:5}"; do
  case $arg in
    -num_refinement_stage=*)
      num_refinement_stage="${arg#*=}"
      ;;
    -stage_layers=*)
      stage_layers="${arg#*=}"
      ;;
    -topks=*)
      topks="${arg#*=}"
      ;;
    -inference_model=*)
      inference_model="${arg#*=}"
      ;;
  esac
done


dir_name=$(echo $dir | tr '/' '-' | tr '.' 'dot')
PREFIX_PATH=/lustre/fsw/portfolios/av/projects/av_research/users/shiyil/yaowenh

submit_job \
    --gpu 8 \
    --tasks_per_node 1 \
    --nodes 1 \
    -n "eval-navsim_ssl-$dir_name-${epoch}epoch" \
    --image $PREFIX_PATH/container_images/ywh-navsim.sqsh \
    --logroot $PREFIX_PATH/slurm_logs/navsim_v2/evaluation-$agent \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition \
    --account av_research \
    -c ". $PREFIX_PATH/pre-navsim_v2.sh; bash scripts/drivesuprim/evaluation/eval.sh \
        $epoch $dir $num_refinement_stage $stage_layers $topks $agent $inference_model"


: '
usage:
bash scripts/slurm_bash/drivesuprim/evaluation/eval-multi_stage.sh \
    5 training/drivesuprim_prev_ckpts/stage_layers_3-topks_256-vit \
    drivesuprim_agent_vit interactive \
    -num_refinement_stage=1 -stage_layers=3 -topks=256 -inference_model=teacher
'
