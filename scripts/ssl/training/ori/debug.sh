agent="hydra_img_vit_ssl"
bs=8
lr=0.000025
epoch=20
config="competition_training"
rot=0
trans=0
va=0
probability=0.0
dir=training/ssl/ori/lr_baseline-debug

command_string="python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=false \
    agent=$agent \
    experiment_name=$dir \
    split=trainval \
    scene_filter=navtrain_debug \
    dataloader.params.batch_size=$bs \
    ~trainer.params.strategy \
    trainer.params.max_epochs=$epoch \
    agent.config.ckpt_path=$dir \
    agent.lr=$lr \
    agent.config.ego_perturb.mode=load_from_offline \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_$rot-trans_$trans-va_$va-p_$probability.json \
    agent.config.ego_perturb.rotation.enable=false \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=${rot} \
    agent.config.only_ori_input=true \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_debug/navtrain_debug.pkl \
    cache_path=null
"

echo "--- COMMAND ---"
echo $command_string
echo "\n\n"

eval $command_string
