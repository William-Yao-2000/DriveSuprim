TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    +debug=false \
    agent=transfuser_agent \
    experiment_name=training/transfuser_agent \
    ~trainer.params.strategy \
    trainer.params.limit_val_batches=0.2 \
    train_test_split=$TRAIN_TEST_SPLIT \
    +resume_ckpt_path="${NAVSIM_EXP_ROOT}/training/transfuser_agent/epoch\=23-step\=4008.ckpt"
    cache_path=null


: '
debug:

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 \
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    +debug=true \
    agent=transfuser_agent \
    experiment_name=debug \
    \~trainer.params.strategy \
    train_test_split=navtrain_debug \
    cache_path=null
'