#!/bin/bash
bash_file=$1
partition=$2

for arg in "${@:3}"; do
  case $arg in
    -only_ori_input=*)
      only_ori_input="${arg#*=}"
      ;;
    -ban_soft_label_loss=*)
      ban_soft_label_loss="${arg#*=}"
      ;;
  esac
done

dir_name=$(echo $bash_file-multi_stage_lab-r34 | tr '/' '-' | tr '.' 'dot')

if [ "$only_ori_input" = "true" ]; then
    dir_name="$dir_name-only_ori_input"
fi

if [ "$ban_soft_label_loss" = "true" ]; then
    dir_name="$dir_name-ban_soft_label_loss"
fi

for epoch in $(seq 2 3 10)
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
        --duration 3.5 \
        --dependent_clones 0 \
        --partition $partition \
        --account av_research \
        -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; bash $bash_file $only_ori_input $ban_soft_label_loss $epoch"

    sleep 3.5h

done

: '
usage:
bash scripts/slurm_bash/ssl/training/teacher_student/multi_stage/labs-r34/ori_input-soft_label_loss-resume_3ep.sh \
    scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/labs-r34/ori_input-soft_label_loss-resume.sh \
    interactive \
    -only_ori_input=true -ban_soft_label_loss=true
'
