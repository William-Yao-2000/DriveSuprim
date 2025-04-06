#!/bin/bash

#agent=hydra_plus
#dir=hydra_plus_video
#bs=22
#nodes=3
#epochs=20

agent=dp_v2
dir=dp_ctrl_v2
bs=32
nodes=2
epochs=100

submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --account av_research \
    --nodes ${nodes} \
    -n ${dir} \
    --image /lustre/fsw/portfolios/av/users/zhenxinl/navsim_workspace/lzx-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/zhenxinl/navsim_workspace/slurm_logs \
    --email_mode never \
    --duration 4 \
    --dependent_clones 2 \
    --partition interactive \
    -c ". slurm/pre.sh; bash slurm/train/train_dp.sh ${agent} ${dir} ${bs} ${epochs}"