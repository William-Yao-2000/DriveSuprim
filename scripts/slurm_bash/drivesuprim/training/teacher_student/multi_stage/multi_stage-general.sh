#!/bin/bash
bash_file=$1
agent=$2
num_refinement_stage=$3
stage_layers=$4
topks=$5
partition=$6

dir_name=$(echo $bash_file-$agent-$stage_layers-$topks | tr '/' '-' | tr '.' 'dot')

PREFIX_PATH=/lustre/fsw/portfolios/av/projects/av_research/users/shiyil/yaowenh

submit_job \
    --gpu 8 \
    --tasks_per_node 1 \
    --nodes 1 \
    -n "$dir_name" \
    --image $PREFIX_PATH/container_images/ywh-navsim.sqsh \
    --logroot $PREFIX_PATH/slurm_logs/navsim_v2/training-$agent \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition \
    --account av_research \
    -c ". $PREFIX_PATH/pre-navsim_v2.sh; bash $bash_file $agent $num_refinement_stage $stage_layers $topks"


: '
usage:
bash scripts/slurm_bash/drivesuprim/training/teacher_student/multi_stage/multi_stage-general.sh \
    scripts/drivesuprim/training/rot_30-p_0.5/train.sh \
    drivesuprim_agent_r34 1 3 256 \
    interactive
'
