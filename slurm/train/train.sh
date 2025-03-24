agent=$1
dir=$1

bs=32
lr=0.0002
epoch=20
config="competition_training"


command_string="
    MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        --config-name $config \
        trainer.params.num_nodes=${NUM_NODES} \
        agent=$agent \
        agent.pdm_gt_path=vocab_score_full_16384_navtrain_v2/navtrain.pkl \
        experiment_name=$dir \
        split=trainval \
        scene_filter=navtrain \
        dataloader.params.batch_size=$bs \
        ~trainer.params.strategy \
        trainer.params.max_epochs=$epoch \
        agent.config.ckpt_path=$dir \
        agent.lr=$lr \
        cache_path=null
"

echo "--- COMMAND ---"
echo $command_string
echo "\n\n"

cd $NAVSIM_DEVKIT_ROOT
pwd

eval $command_string
