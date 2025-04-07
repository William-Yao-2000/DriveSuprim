#!/bin/bash
partition_=$1

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes 1 \
    -n "navsim_ssl-teacher_student-rot_30-trans_0-va_0-p_0.5-not_rotation-mask" \
    --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/training \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition_ \
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre.sh; bash scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/not_rotation-mask.sh"
