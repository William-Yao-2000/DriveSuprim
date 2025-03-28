#!/bin/bash

agent=$1
dir=$2

# 2nodes, bs=32
# 3nodes, bs=22
bs=$3
lr=0.0002
max_epochs=20

config="competition_training"
ckpt_dir="${NAVSIM_EXP_ROOT}/${dir}"

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO

# Function to execute training from scratch
run_train_from_scratch() {
    command_string="
        MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
        python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
            --config-name $config \
            trainer.params.num_nodes=${NUM_NODES} \
            agent=$agent \
            agent.pdm_gt_path=vocab_score_full_16384_navtrain_v2ep/navtrain.pkl \
            experiment_name=$dir \
            train_test_split=navtrain \
            dataloader.params.batch_size=$bs \
            ~trainer.params.strategy \
            trainer.params.max_epochs=${max_epochs} \
            trainer.params.precision=32 \
            agent.config.ckpt_path=$dir \
            agent.lr=$lr \
            cache_path=null
    "
    echo "--- COMMAND ---"
    echo "$command_string"
    echo -e "\n\n"

    cd $NAVSIM_DEVKIT_ROOT
    pwd

    eval "$command_string"
}

# Function to resume training from latest checkpoint
resume_training() {
    local latest_epoch=$1

    padded_ckpt_epoch=$(printf "%02d" $latest_epoch)
    resume="epoch${padded_ckpt_epoch}.ckpt"

    command_string="
        MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
        python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
            --config-name $config \
            trainer.params.num_nodes=${NUM_NODES} \
            +resume_ckpt_path='${NAVSIM_EXP_ROOT}/$dir/$resume' \
            agent=$agent \
            agent.pdm_gt_path=vocab_score_full_16384_navtrain_v2ep/navtrain.pkl \
            experiment_name=$dir \
            train_test_split=navtrain \
            dataloader.params.batch_size=$bs \
            ~trainer.params.strategy \
            trainer.params.max_epochs=${max_epochs} \
            trainer.params.precision=32 \
            agent.config.ckpt_path=$dir \
            agent.lr=$lr \
            cache_path=null
    "
    echo "--- COMMAND ---"
    echo "$command_string"
    echo -e "\n\n"

    cd $NAVSIM_DEVKIT_ROOT
    pwd

    eval "$command_string"
}

# Main logic
if [ ! -d "$ckpt_dir" ]; then
    echo "Directory $ckpt_dir does not exist. Training from scratch..."
    run_train_from_scratch
else
    echo "Checking for checkpoints in $ckpt_dir..."
    cd $ckpt_dir
    for file in epoch=*-step=*.ckpt; do
      epoch=$(echo $file | sed -n 's/.*epoch=\([0-9][0-9]\).*/\1/p')
      new_filename="epoch${epoch}.ckpt"
      mv "$file" "$new_filename"
    done
    cd $NAVSIM_NAVSIM_DEVKIT_ROOT


    shopt -s nullglob
    ckpt_files=("$ckpt_dir"/epoch*.ckpt)
    shopt -u nullglob

    if [ ${#ckpt_files[@]} -eq 0 ]; then
        echo "No checkpoint found. Training from scratch..."
        run_train_from_scratch
    else
        echo "Found ${#ckpt_files[@]} checkpoint(s). Detecting latest epoch..."
        latest_epoch=-1
        for file in "${ckpt_files[@]}"; do
            base_file=$(basename "$file")
            # Extract epoch number from filename format: epochXX.ckpt
            epoch_str=${base_file#epoch}   # Remove 'epoch' prefix
            epoch_str=${epoch_str%.ckpt}    # Remove '.ckpt' suffix
            epoch_num=$((10#$epoch_str))    # Convert to base-10 number

            if (( epoch_num > latest_epoch )); then
                latest_epoch=$epoch_num
            fi
        done

        if (( latest_epoch >= 0 )); then
            echo "Resuming from epoch $latest_epoch..."
            resume_training $latest_epoch
        else
            echo "Invalid checkpoint format. Training from scratch..."
            run_train_from_scratch
        fi
    fi
fi