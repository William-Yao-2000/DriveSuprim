#!/bin/bash
bash_file=$1
partition=$2
soft_label_imi_diff_thresh=$3

dir_name=$(echo $bash_file-multi_stage_labs-r34-imi_thresh_$soft_label_imi_diff_thresh | tr '/' '-' | tr '.' 'dot')

PREFIX_PATH=/lustre/fsw/portfolios/av/projects/av_research/users/shiyil/yaowenh

for epoch in $(seq 0 1 10)
do
    echo $epoch
    submit_job \
        --gpu 8 \
        --tasks_per_node 1 \
        --nodes 1 \
        -n "$dir_name--$epoch" \
        --image $PREFIX_PATH/container_images/ywh-navsim.sqsh \
        --logroot $PREFIX_PATH/slurm_logs/navsim_v2/training \
        --email_mode never \
        --duration 3 \
        --dependent_clones 0 \
        --partition $partition \
        --account av_research \
        -c ". $PREFIX_PATH/pre-navsim_v2.sh; bash $bash_file $soft_label_imi_diff_thresh $epoch"
    
    sleep 2.9h

done


: '
usage:
bash scripts/slurm_bash/ssl/training/teacher_student/multi_stage/labs-r34/imi_diff_thresh-resume_1ep.sh \
    scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs-r34/imi_diff_thresh-resume.sh \
    interactive 0.5
'
