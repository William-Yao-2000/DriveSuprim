#!/bin/bash

# 1: split (navtrain/navtest) 2: part (sub1...sub100)

submit_job \
    --gpu 1 \
    --tasks_per_node 1 \
    --nodes 1 \
    -n "zx_gen" \
    --image /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/lzx-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/slurm_logs \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    -c ". slurm/pre.sh; bash slurm/gen/gen.sh $1 $2"
