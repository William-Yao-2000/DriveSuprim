#!/bin/bash
bash_file=$1
num_refinement_stage=$2
stage_layers=$3
topks=$4
partition=$5

dir_name=$(echo $bash_file-$stage_layers-$topks | tr '/' '-' | tr '.' 'dot')

for epoch in $(seq 3 4 7)
do
    echo $epoch
    submit_job \
        --gpu 8 \
        --tasks_per_node 8 \
        --nodes 1 \
        -n "$dir_name--$epoch" \
        --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
        --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/navsim_v2/training \
        --email_mode never \
        --duration 4 \
        --dependent_clones 0 \
        --partition $partition \
        --account av_research \
        -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash $bash_file $num_refinement_stage $stage_layers $topks $epoch"

    sleep 4.2h

done


: '
usage:
bash scripts/slurm_bash/ssl/training/teacher_student/multi_stage/multi_stage-general-resume_4ep.sh \
    scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/multi_stage-general-resume.sh \
    1 3 256 \
    interactive
'
