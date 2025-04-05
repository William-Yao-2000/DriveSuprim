export agent=dp
export res="2048x512"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
--config-name competition_training \
agent=${agent} \
agent.pdm_gt_path=1234 \
worker.threads_per_node=16 \
train_test_split=navtrain \
experiment_name=debug \
cache_path=${NAVSIM_EXP_ROOT}/navtrain_${agent}_img${res}_cache
