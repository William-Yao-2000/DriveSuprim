#!/bin/bash

agent=hydra_plus
dir=hydra_plus_16mixed
bs=32
nodes=2
repeat=1

#submit_job \
#    --gpu 8 \
#    --tasks_per_node 8 \
#    --nodes ${nodes} \
#    -n "zxtrain2" \
#    --image /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/lzx-navsim.sqsh \
#    --logroot /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/slurm_logs \
#    --email_mode never \
#    --duration 4 \
#    --dependent_clones 2 \
#    -c ". slurm/pre.sh; bash slurm/train/train_auto.sh ${agent} ${dir} ${bs}"

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes ${nodes} \
    -n "zxtrain2" \
    --image /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/lzx-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/slurm_logs \
    --email_mode never \
    --duration 4 \
    --dependent_clones $repeat \
    -c ". slurm/pre.sh; bash slurm/train/train_auto.sh ${agent} ${dir} ${bs}"