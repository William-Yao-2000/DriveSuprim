#!/bin/bash
bash_file=$1
partition=$2

for arg in "${@:3}"; do
  case $arg in
    -change_loss_weight=*)
      change_loss_weight="${arg#*=}"
      ;;
    -use_imi_learning_in_refinement=*)
      use_imi_learning_in_refinement="${arg#*=}"
      ;;
  esac
done

dir_name=$(echo $bash_file-multi_stage_lab | tr '/' '-' | tr '.' 'dot')

if [ "$change_loss_weight" = "true" ]; then
    dir_name="$dir_name-change_loss_weight"
fi

if [ "$use_imi_learning_in_refinement" = "true" ]; then
    dir_name="$dir_name-use_imi_learning_in_refinement"
fi

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes 1 \
    -n "$dir_name" \
    --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/navsim_v2/training \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition \
    --account av_research \
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash $bash_file $change_loss_weight $use_imi_learning_in_refinement"


: '
usage:
bash scripts/slurm_bash/ssl/training/teacher_student/multi_stage/labs/loss_weight-imi_learning.sh \
    scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs/loss_weight-imi_learning.sh \
    interactive_singlenode \
    -change_loss_weight=false -use_imi_learning_in_refinement=true
'
