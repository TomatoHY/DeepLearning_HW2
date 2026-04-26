# Task 1: CIFAR-10 图像分类

使用ResNet18在CIFAR-10数据集上进行图像分类。

## 环境配置

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py \
    --data_path /mnt/data/kw/hy/projects/course/DL/hw2/datasets/cifar10 \
    --save_dir ./checkpoints \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.1 \
    --scheduler cosine \
    --augment
```

### 主要参数说明

- `--data_path`: CIFAR-10数据集路径
- `--save_dir`: 模型保存目录
- `--batch_size`: 批次大小（默认128）
- `--epochs`: 训练轮数（默认200）
- `--lr`: 初始学习率（默认0.1）
- `--scheduler`: 学习率调度器（cosine/step/none）
- `--augment`: 使用数据增强
- `--pretrained`: 使用预训练模型

## 测试

```bash
python test.py \
    --data_path /mnt/data/kw/hy/projects/course/DL/hw2/datasets/cifar10 \
    --checkpoint ./checkpoints/best_model.pth \
    --output_path ./confusion_matrix.png
```

## 模型架构

- 基础模型：ResNet18
- 修改：
  - 第一层卷积：7x7 → 3x3（适配32x32小图像）
  - 移除MaxPooling层
  - 最后全连接层：1000 → 10类

## 数据增强

- RandomCrop(32, padding=4)
- RandomHorizontalFlip
- ColorJitter
- Normalize

## 预期结果

- 训练准确率：~95%
- 测试准确率：~93-94%
