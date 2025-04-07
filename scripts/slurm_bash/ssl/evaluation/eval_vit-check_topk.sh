#!/bin/bash
epoch=$1
dir=$2
num_top_k=$3
partition=$4

dir_name=$(echo $dir | tr '/' '-' | tr '.' 'dot')

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes 1 \
    -n "eval-navsim_ssl-$dir_name-${epoch}epoch-check_k_${num_top_k}" \
    --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/evaluation/ \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    --partition $partition \
    -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre.sh; bash scripts/ssl/evaluation/eval_vit-check_topk.sh $epoch $dir $num_top_k"
