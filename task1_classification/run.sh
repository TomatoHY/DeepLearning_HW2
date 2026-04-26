#!/bin/bash

# Task 1: CIFAR-10 Classification Training Script

# 设置数据路径
DATA_PATH="/mnt/data/kw/hy/projects/course/DL/hw2/datasets/cifar10"
SAVE_DIR="./checkpoints"

# 训练
echo "Starting training..."
python train.py \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.1 \
    --scheduler cosine \
    --augment \
    --num_workers 4

# 测试
echo "Starting testing..."
python test.py \
    --data_path $DATA_PATH \
    --checkpoint $SAVE_DIR/best_model.pth \
    --output_path ./confusion_matrix.png \
    --batch_size 128 \
    --num_workers 4

echo "Task 1 completed!"
