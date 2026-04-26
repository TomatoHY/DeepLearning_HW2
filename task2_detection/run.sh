#!/bin/bash

# Task 2: COCO Object Detection Training Script

# 设置数据路径
DATA_PATH="/mnt/data/kw/hy/projects/course/DL/hw2/datasets/coco"
SAVE_DIR="./checkpoints"

# 训练
echo "Starting training..."
python train.py \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --batch_size 4 \
    --epochs 12 \
    --lr 0.005 \
    --num_workers 4 \
    --print_freq 100

# 评估
echo "Starting evaluation..."
python evaluate.py \
    --data_path $DATA_PATH \
    --checkpoint $SAVE_DIR/best_model.pth \
    --batch_size 4 \
    --num_workers 4

echo "Task 2 completed!"
