#!/bin/sh
OUTPUT_DIR='output/pretrain_mae_base_patch16_224'
DATA_PATH='/home/featurize/data'

# batch_size can be adjusted according to the graphics card
#rtx3060 almost 3min per epoch
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 entry.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 128 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 160 \
        --num_workers 5 \
        --output_dir ${OUTPUT_DIR}