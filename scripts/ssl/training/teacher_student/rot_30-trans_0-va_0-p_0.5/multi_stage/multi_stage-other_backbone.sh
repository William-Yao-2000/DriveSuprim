bs=8
lr=0.000075
epoch=8
config="competition_training"
rot=30
trans=0
va=0
probability=0.5

agent=$1
num_refinement_stage=$2
stage_layers=$3
topks=$4

dir=training/ssl/teacher_student/rot_$rot-trans_$trans-va_$va-p_$probability/multi_stage/stage_layers_$stage_layers-topks_$topks-$agent


command_string="python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=false \
    agent=$agent \
    experiment_name=$dir \
    split=trainval \
    train_test_split=navtrain \
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
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_final/navtrain.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_$rot-trans_$trans-va_$va-p_$probability-ensemble/split_pickles \
    cache_path=null
"

echo "--- COMMAND ---"
echo $command_string
echo "\n\n"

eval $command_string


: '
usage:
bash scripts/ssl/training/teacher_student/rot_30-trans_0-va_0-p_0.5/multi_stage/multi_stage-other_backbone-general.sh \
  hydra_img_r50_ssl 1 3 256
'