#!/bin/bash
epoch=$1
dir=$2
partition=$3


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
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash scripts/ssl/evaluation/labs-r34/eval_r34-multi_stage-5_camera.sh $epoch $dir"


: '
usage:
bash scripts/slurm_bash/ssl/evaluation/labs-r34/eval_r34-multi_stage-5_camera.sh \
    9 training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs-r34/stage_layers_3-topks_256-5_camera \
    interactive_singlenode
'
