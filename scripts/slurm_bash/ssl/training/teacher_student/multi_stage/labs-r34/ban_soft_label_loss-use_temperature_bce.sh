#!/bin/bash
bash_file=$1
partition=$2

dir_name=$(echo $bash_file-multi_stage_lab-r34-ban_soft_label_loss-use_temperature_bce | tr '/' '-' | tr '.' 'dot')

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
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash $bash_file"


: '
usage:
bash scripts/slurm_bash/ssl/training/teacher_student/multi_stage/labs-r34/ban_soft_label_loss-use_temperature_bce.sh \
  scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs-r34/ban_soft_label_loss-use_temperature_bce.sh \
  interactive
'
