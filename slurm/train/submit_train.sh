#!/bin/bash

agent=hydra_plus
dir=hydra_plus_v2ep
bs=22
nodes=3

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --account av_research \
    --nodes ${nodes} \
    --partition interactive \
    -n "zxtrain2" \
    --image /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/lzx-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/slurm_logs \
    --email_mode never \
    --duration 4 \
    --dependent_clones 2 \
    -c ". slurm/pre.sh; bash slurm/train/train_auto_fp32.sh ${agent} ${dir} ${bs}"