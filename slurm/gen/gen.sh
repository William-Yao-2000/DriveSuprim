export split=$1
export part=$2
python $NAVSIM_DEVKIT_ROOT/navsim/agents/tools/gen_vocab_score.py \
train_test_split=${split}_${part} \
experiment_name=debug \
worker.threads_per_node=128 \
+save_name=${split}_${part}_v2 \
metric_cache_path=/lustre/fsw/portfolios/av/users/shiyil/zxli/navsim_workspace/exp2/${split}_metric_cache

