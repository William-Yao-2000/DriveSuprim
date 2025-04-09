#!/bin/bash
start_=$1
end_=$2
seed_=$3
partition_=$4

for idx in $(seq $start_ $end_)
do
    submit_job \
        --gpu 1 \
        --tasks_per_node 1 \
        --nodes 1 \
        -n "gen_full_vocab_pdm_score_navtrain_sub${idx}_seed_$seed_" \
        --image /lustre/fsw/portfolios/av/users/shiyil/yaowenh/container_images/ywh-navsim.sqsh \
        --logroot /lustre/fsw/portfolios/av/users/shiyil/yaowenh/slurm_logs/gen_full_vocab_pdm_score \
        --email_mode never \
        --duration 4 \
        --dependent_clones 0 \
        --account av_research \
        --partition $partition_ \
        -c ". /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre-navsim_v2.sh; \
        bash scripts/ssl/gen_full_score_aug/gen_training_full_score_aug_subset-seeds.sh navtrain_sub$idx $seed_"
    sleep 3
done