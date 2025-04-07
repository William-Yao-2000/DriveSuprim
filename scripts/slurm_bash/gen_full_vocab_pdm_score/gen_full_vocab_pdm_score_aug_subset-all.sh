#!/bin/bash
cd /lustre/fsw/portfolios/av/users/shiyil/yaowenh/proj/navsim_workspace/robust_navsim
git pull

for idx in $(seq 1 32)
do
    submit_job \
        --gpu 1 \
        --tasks_per_node 1 \
        --nodes 1 \
        -n "gen_full_vocab_pdm_score_navtrain_new_sub$idx" \
        --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
        --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/gen_full_vocab_pdm_score \
        --email_mode never \
        --duration 4 \
        --dependent_clones 0 \
        -c "bash scripts/slurm_bash/gen_full_vocab_pdm_score/gen_full_vocab_pdm_score_aug_subset.sh navtrain_new_sub$idx"
    sleep 3
done