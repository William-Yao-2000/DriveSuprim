TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=transfuser_agent \
    experiment_name=training/transfuser_agent \
    ~trainer.params.strategy \
    trainer.params.limit_val_batches=0.05 \
    train_test_split=$TRAIN_TEST_SPLIT \
