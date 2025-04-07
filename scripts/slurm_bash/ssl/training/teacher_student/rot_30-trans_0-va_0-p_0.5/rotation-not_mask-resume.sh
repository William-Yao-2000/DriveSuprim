#!/bin/bash

partition=$1

for epoch in $(seq 1 2 17)
do
    echo $epoch
    submit_job \
        --gpu 8 \
        --tasks_per_node 8 \
        --nodes 1 \
        -n "navsim_ssl-teacher_student-rot_30-trans_0-va_0-p_0.5-ensemble_3-lr3x" \
        --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
        --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/training \
        --email_mode never \
        --duration 4 \
        --dependent_clones 0 \
        --partition $partition \
        -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre.sh; bash scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/rotation-not_mask-resume.sh $epoch"
    
    sleep 4.2h
done
