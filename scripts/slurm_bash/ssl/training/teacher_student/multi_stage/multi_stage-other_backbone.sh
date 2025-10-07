#!/bin/bash
bash_file=$1
agent=$2
num_refinement_stage=$3
stage_layers=$4
topks=$5
partition=$6

dir_name=$(echo $bash_file-$stage_layers-$topks-$agent | tr '/' '-' | tr '.' 'dot')

PREFIX_PATH=/lustre/fsw/portfolios/av/projects/av_research/users/shiyil/yaowenh

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes 1 \
    -n "$dir_name" \
    --image $PREFIX_PATH/container_images/ywh-navsim.sqsh \
    --logroot $PREFIX_PATH/slurm_logs/navsim_v2/training \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition \
    --account av_research \
    -c ". $PREFIX_PATH/pre-navsim_v2.sh; bash $bash_file $agent $num_refinement_stage $stage_layers $topks"


: '
usage:
bash scripts/slurm_bash/ssl/training/teacher_student/multi_stage/multi_stage-other_backbone.sh \
    scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/multi_stage-other_backbone.sh \
    hydra_img_sptr_ssl 1 3 256 \
    interactive
'
