TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    +debug=false \
    experiment_name=training/ego_mlp_agent \
    ~trainer.params.strategy \
    trainer.params.max_epochs=50 \
    trainer.params.limit_val_batches=0.2 \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=null


: '
debug:

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    +debug=true \
    experiment_name=debug \
    \~trainer.params.strategy \
    trainer.params.max_epochs=50 \
    train_test_split=navtrain_debug \
    cache_path=null
'