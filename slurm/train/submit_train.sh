#!/bin/bash

agent=hydra_plus

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes 4 \
    -n "zxtrain2" \
    --image /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/lzx-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/slurm_logs \
    --email_mode never \
    --duration 4 \
    --dependent_clones 1 \
    --partition polar \
    -c ". slurm/pre.sh; bash slurm/train/train_auto.sh ${agent}"
