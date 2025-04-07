#!/bin/bash
scene_filter=$1
. /lustre/fsw/portfolios/av/users/shiyil/yaowenh/pre.sh

bash scripts/ssl/gen_full_score_aug/gen_training_full_score_aug_subset-seeds.sh $scene_filter