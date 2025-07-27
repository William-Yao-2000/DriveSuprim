#!/bin/bash
bash_file=$1
partition=$2

dir_name=$(echo $bash_file-multi_stage_lab-r34-single_stage | tr '/' '-' | tr '.' 'dot')

for epoch in $(seq 5 1 10)
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
        --duration 3.1 \
        --dependent_clones 0 \
        --partition $partition \
        --account av_research \
        -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash $bash_file $epoch"

    sleep 3.2h

done


: '
usage:
bash scripts/slurm_bash/ssl/training/teacher_student/multi_stage/labs-r34/single_stage-resume_1ep.sh \
    scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs-r34/single_stage-resume.sh \
    interactive_singlenode
'
