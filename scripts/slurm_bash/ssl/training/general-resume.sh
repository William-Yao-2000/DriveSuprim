#!/bin/bash

bash_file=$1
partition=$2

dir_name=$(echo $bash_file | tr '/' '-' | tr '.' 'dot')

for epoch in $(seq 1 2 7)
do
    echo $epoch
    submit_job \
        --gpu 8 \
        --tasks_per_node 8 \
        --nodes 1 \
        -n "$dir_name--$epoch" \
        --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
        --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/training \
        --email_mode never \
        --duration 4 \
        --dependent_clones 0 \
        --partition $partition \
        -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre.sh; bash $bash_file $epoch"
    
    sleep 4.2h
done


# usage: bash scripts/slurm_bash/ssl/training/general-resume.sh scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/two_stage-resume.sh interactive