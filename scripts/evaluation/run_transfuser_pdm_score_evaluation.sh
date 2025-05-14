ckpt_epoch=19

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="${NAVSIM_EXP_ROOT}/training/transfuser_agent/epoch\=${ckpt_epoch}-step\=3340.ckpt"
CACHE_PATH="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage_gpu.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=transfuser_agent \
    worker=single_machine_thread_pool \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=training/transfuser_agent/test-one_stage-${ckpt_epoch}ep \
    metric_cache_path=$CACHE_PATH
