agent="hydra_img_vit_ssl"
bs=8
lr=0.000075
epoch=20
config="competition_training"
rot=30
trans=0
va=0
probability=0.5
soft_thre=1.0
dir=training/ssl/teacher_student/rot_$rot-trans_$trans-va_$va-p_$probability/soft_th_1dot0

ckpt_epoch=$1
# Format epoch with leading zero
padded_ckpt_epoch=$(printf "%02d" $ckpt_epoch)

# Calculate step from epoch (1330 steps per epoch)
step=$((($ckpt_epoch + 1) * 1330))

resume="epoch\=${padded_ckpt_epoch}-step\=${step}.ckpt"

if [ -z "$resume" ]; then
    echo -e "Wrong! You need to provide the resume model name!"
    exit 1
fi

command_string="python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=false \
    agent=$agent \
    experiment_name=$dir \
    +resume_ckpt_path='$NAVSIM_EXP_ROOT/$dir/$resume' \
    split=trainval \
    scene_filter=navtrain \
    dataloader.params.batch_size=$bs \
    ~trainer.params.strategy \
    trainer.params.max_epochs=$epoch \
    trainer.params.limit_val_batches=0.05 \
    agent.config.ckpt_path=$dir \
    agent.lr=$lr \
    agent.config.ego_perturb.mode=load_from_offline \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_$rot-trans_$trans-va_$va-p_$probability-ensemble.json \
    agent.config.ego_perturb.rotation.enable=true \
    agent.config.ego_perturb.rotation.offline_aug_angle_boundary=$rot \
    agent.config.student_rotation_ensemble=3 \
    agent.config.soft_label_diff_thresh=$soft_thre \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_full_8192_navtrain/navtrain.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_$rot-trans_$trans-va_$va-p_$probability-ensemble/split_pickles \
    cache_path=null
"

echo "--- COMMAND ---"
echo $command_string
echo "\n\n"

eval $command_string
