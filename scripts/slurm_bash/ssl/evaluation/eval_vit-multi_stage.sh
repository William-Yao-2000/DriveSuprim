#!/bin/bash
epoch=$1
dir=$2
partition=$3

# Parse additional arguments
# TODO: use dir to get these parameters
for arg in "${@:4}"; do
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
  esac
done


dir_name=$(echo $dir | tr '/' '-' | tr '.' 'dot')

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes 1 \
    -n "eval-navsim_ssl-$dir_name-${epoch}epoch" \
    --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/evaluation/ \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition \
    --account av_research \
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash scripts/ssl/evaluation/eval_vit-multi_stage.sh $epoch $dir $num_refinement_stage $stage_layers $topks"
