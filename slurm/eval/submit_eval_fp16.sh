#!/bin/bash

# $1, $2, $3: epoch, dir, agent
# bash slurm/submit_eval.sh 16 vadv2_10k vadv2_10k
agent=hydra_plus
dir=hydra_plus_16mixed

for i in {10..19}; do
  submit_job \
    --gpu 8 \
    --tasks_per_node 8 \
    --nodes 1 \
    -n "zx_eval_fp16" \
    --image /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/lzx-navsim.sqsh \
    --logroot /lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/slurm_logs \
    --email_mode never \
    --duration 4 \
    --dependent_clones 0 \
    -c ". slurm/pre.sh; bash slurm/eval/eval_fp16.sh ${i} $dir $agent"
done

