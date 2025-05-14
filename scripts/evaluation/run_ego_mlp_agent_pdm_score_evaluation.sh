ckpt_epoch=19

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="${NAVSIM_EXP_ROOT}/training/ego_mlp_agent/epoch\=${ckpt_epoch}-step\=3340.ckpt"
CACHE_PATH="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=ego_status_mlp_agent \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=training/ego_mlp_agent/test-one_stage-${ckpt_epoch}ep \
    metric_cache_path=$CACHE_PATH 
