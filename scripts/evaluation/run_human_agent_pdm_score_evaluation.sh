TRAIN_TEST_SPLIT=navtest
CACHE_PATH="${NAVSIM_EXP_ROOT}/metric_cache/test/ori"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=human_agent \
    experiment_name=training/human_agent/test-one_stage \
    traffic_agents_policy=non_reactive \
    metric_cache_path=$CACHE_PATH \
