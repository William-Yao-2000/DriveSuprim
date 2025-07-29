#!/bin/bash
epoch=$1
dir=$2
inference_model=$3
partition=$4


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
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash scripts/ssl/evaluation/labs-r34/eval_r34-single_stage.sh $epoch $dir $inference_model" \


: '
usage:
bash scripts/slurm_bash/ssl/evaluation/labs-r34/eval_r34-single_stage.sh \
  8 training/ssl/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs-r34/single_stage \
  teacher interactive_singlenode
'
