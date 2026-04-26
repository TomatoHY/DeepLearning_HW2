# Task 2: COCO 目标检测

使用Faster R-CNN在COCO数据集上进行目标检测。

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据集准备

COCO数据集应包含以下结构：
```
coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## 训练

```bash
python train.py \
    --data_path /mnt/data/kw/hy/projects/course/DL/hw2/datasets/coco \
    --save_dir ./checkpoints \
    --batch_size 4 \
    --epochs 12 \
    --lr 0.005
```

### 主要参数说明

- `--data_path`: COCO数据集路径
- `--save_dir`: 模型保存目录
- `--batch_size`: 批次大小（默认4，根据GPU内存调整）
- `--epochs`: 训练轮数（默认12）
- `--lr`: 初始学习率（默认0.005）
- `--print_freq`: 打印频率

## 评估

```bash
python evaluate.py \
    --data_path /mnt/data/kw/hy/projects/course/DL/hw2/datasets/coco \
    --checkpoint ./checkpoints/best_model.pth \
    --visualize \
    --num_vis 10
```

### 评估参数

- `--checkpoint`: 模型检查点路径
- `--visualize`: 生成可视化结果
- `--num_vis`: 可视化图像数量

## 模型架构

- 基础模型：Faster R-CNN with ResNet-50-FPN backbone
- 预训练：ImageNet预训练权重
- 类别数：80个COCO类别 + 背景

## 训练策略

- 优化器：SGD (momentum=0.9, weight_decay=0.0005)
- 学习率调度：StepLR (step_size=3, gamma=0.1)
- 数据增强：默认使用torchvision的检测增强

## 评估指标

- mAP@0.5:0.95
- mAP@0.5
- mAP@0.75
- mAP (small/medium/large objects)

## 预期结果

- mAP@0.5:0.95: ~0.37 (12 epochs)
- 完整训练(26 epochs): ~0.40+
