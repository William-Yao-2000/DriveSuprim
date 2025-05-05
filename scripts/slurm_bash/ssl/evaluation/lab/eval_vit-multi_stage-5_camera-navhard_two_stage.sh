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
    -use_first_stage_traj_in_infer=*)
      use_first_stage_traj_in_infer="${arg#*=}"
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
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash scripts/ssl/evaluation/lab/eval_vit-multi_stage-5_camera-navhard_two_stage.sh $epoch $dir $num_refinement_stage $stage_layers $topks $use_first_stage_traj_in_infer"


: '
usage:
bash scripts/slurm_bash/ssl/evaluation/lab/eval_vit-multi_stage-5_camera-navhard_two_stage.sh \
    5 training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs/stage_layers_3-topks_256-5_camera \
    interactive \
    -num_refinement_stage=1 -stage_layers=3 -topks=256 -use_first_stage_traj_in_infer=true
'
