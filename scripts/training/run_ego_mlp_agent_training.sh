TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=training/ego_mlp_agent \
    ~trainer.params.strategy \
    trainer.params.max_epochs=50 \
    trainer.params.limit_val_batches=0.05 \
    train_test_split=$TRAIN_TEST_SPLIT \
