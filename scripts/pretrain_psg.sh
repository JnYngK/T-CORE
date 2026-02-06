#!/usr/bin/env bash
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

LMDB_PATH=/data2/flow_dataset.lmdb
ENTRIES_CSV=/path/to/label_example.csv   # your CSV path

torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    base_model/train/train.py \
    --config-file base_model/configs/vits16_psg.yaml \
    --output-dir save/vits16_flow/PSG/ \
    # student.arch=vit_small \
    # student.patch_size=16 \
    # auxiliary.use_auxiliary=true \
    # auxiliary.lr_mul=0.1 \
    # ibot.past_tea_ibot_loss_weight=0.8 \
    # ibot.future_tea_ibot_loss_weight=0.8 \
    # ibot.past_future_MSE_loss_weight=20 \
    # optim.base_lr=0.002 \
    # optim.epochs=400 \
    # optim.warmup_epochs=20 \
    # train.past_offset_range=[0.15,0.25] \
    # train.current_range=[0.3,0.7] \
    # train.future_offset_range=[0.15,0.25] \
    # train.dataset=LMDBFlow \
    # train.dataset_path=LMDBFlow:lmdb=${LMDB_PATH}:entries=${ENTRIES_CSV} \
    # train.batch_size_per_gpu=4 \
    # train.OFFICIAL_EPOCH_LENGTH=936 \
    # evaluation.eval_period_iterations=$((936*20))