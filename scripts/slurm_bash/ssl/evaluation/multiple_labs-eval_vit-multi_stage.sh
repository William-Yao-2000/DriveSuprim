#!/bin/bash
dir=$1
partition=$2

# Parse additional arguments
# TODO: use dir to get these parameters
for arg in "${@:3}"; do
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
    --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/navsim_v2/evaluation/ \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition \
    --account av_research \
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; \
      bash scripts/ssl/evaluation/eval_vit-multi_stage.sh 1 $dir $num_refinement_stage $stage_layers $topks; \
      bash scripts/ssl/evaluation/eval_vit-multi_stage.sh 2 $dir $num_refinement_stage $stage_layers $topks; \
      bash scripts/ssl/evaluation/eval_vit-multi_stage.sh 3 $dir $num_refinement_stage $stage_layers $topks"


: '
usage:
bash scripts/slurm_bash/ssl/evaluation/eval_vit-multi_stage.sh \
    training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/stage_layers_3-topks_256 \
    interactive \
    -num_refinement_stage=1 -stage_layers=3 -topks=256
'
