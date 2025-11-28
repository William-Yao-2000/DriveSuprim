agent=$1
num_refinement_stage=$2
stage_layers=$3
topks=$4

# Validate if agent parameter is in the allowed list
allowed_agents=("drivesuprim_agent_r34" "drivesuprim_agent_r50" "drivesuprim_agent_vit" "drivesuprim_agent_vov")
valid_agent=false
for valid in "${allowed_agents[@]}"; do
  if [ "$agent" == "$valid" ]; then
    valid_agent=true
    break
  fi
done

# If agent parameter is invalid, display error message and exit
if [ "$valid_agent" == false ]; then
  echo "Error: agent parameter must be one of: ${allowed_agents[*]}"
  exit 1
fi

# Set epoch value based on agent type
if [ "$agent" == "drivesuprim_agent_r34" ] || [ "$agent" == "drivesuprim_agent_r50" ]; then
  epoch=10
else
  epoch=8
fi

echo "Using agent: $agent, setting epoch: $epoch\n"

config="competition_training"
bs=8
lr=0.000075
rot=30
probability=0.5


dir=training/$agent/rot_$rot-p_$probability/stage_layers_$stage_layers-topks_$topks


command_string="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_ssl.py \
    +debug=false \
    agent=$agent \
    experiment_name=$dir \
    split=trainval \
    train_test_split=navtrain \
    dataloader.params.batch_size=$bs \
    ~trainer.params.strategy \
    trainer.params.max_epochs=$epoch \
    trainer.params.limit_val_batches=0.1 \
    agent.config.ckpt_path=$dir \
    agent.lr=$lr \
    agent.config.ego_perturb.n_student_rotation_ensemble=3 \
    agent.config.ego_perturb.offline_aug_angle_boundary=$rot \
    agent.config.ego_perturb.offline_aug_file=$NAVSIM_EXP_ROOT/offline_files/training_ego_aug_files/rot_$rot-p_$probability-ensemble.json \
    agent.config.refinement.use_multi_stage=true \
    agent.config.refinement.num_refinement_stage=$num_refinement_stage \
    agent.config.refinement.stage_layers=$stage_layers \
    agent.config.refinement.topks=$topks \
    agent.config.ori_vocab_pdm_score_full_path=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_navtrain_final/navtrain.pkl \
    agent.config.aug_vocab_pdm_score_dir=$NAVSIM_TRAJPDM_ROOT/random_aug/rot_$rot-p_$probability-ensemble/split_pickles \
    cache_path=null
"

echo "--- COMMAND ---"
echo $command_string
echo "\n\n"

torchrun --nproc_per_node=8 --master_port=29500 $command_string


: '
usage:
bash scripts/ssl/training/drivesuprim/rot_30-p_0.5/train.sh \
  drivesuprim_agent_r34 1 3 256
'