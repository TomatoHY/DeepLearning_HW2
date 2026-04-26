#!/bin/bash

# Task 3: U-Net 图像分割训练脚本
# 对比三种损失函数：Cross-Entropy, Dice Loss, Combined (CE+Dice)

DATA_PATH="/mnt/data/kw/hy/projects/course/DL/hw2/datasets/stanford_background"
SAVE_DIR="./checkpoints"
BATCH_SIZE=8
EPOCHS=100
LR=1e-4
IMG_SIZE=256

echo "=========================================="
echo "Task 3: U-Net Image Segmentation"
echo "Dataset: Stanford Background Dataset"
echo "=========================================="

# 创建保存目录
mkdir -p $SAVE_DIR

# 1. 训练 Cross-Entropy Loss
echo ""
echo "=========================================="
echo "Training with Cross-Entropy Loss"
echo "=========================================="
python train.py \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --loss_type ce \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --img_size $IMG_SIZE

# 2. 训练 Dice Loss
echo ""
echo "=========================================="
echo "Training with Dice Loss"
echo "=========================================="
python train.py \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --loss_type dice \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --img_size $IMG_SIZE

# 3. 训练 Combined Loss (CE + Dice)
echo ""
echo "=========================================="
echo "Training with Combined Loss (CE + Dice)"
echo "=========================================="
python train.py \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --loss_type combined \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --img_size $IMG_SIZE

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Evaluating all models..."

# 评估所有模型
for loss_type in ce dice combined; do
    echo ""
    echo "Evaluating $loss_type model..."
    python evaluate.py \
        --data_path $DATA_PATH \
        --checkpoint $SAVE_DIR/best_model_${loss_type}.pth \
        --batch_size $BATCH_SIZE \
        --img_size $IMG_SIZE \
        --visualize \
        --num_vis 10 \
        --save_dir ./visualizations_${loss_type}
done

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
