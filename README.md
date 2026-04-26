# 深度学习作业2 - HW2

本项目包含3个深度学习任务的完整实现。

## 项目结构

```
hw2/
├── download.sh                    # 数据集和模型下载脚本
├── task1_classification/          # 任务1: CIFAR-10图像分类
│   ├── train.py
│   ├── test.py
│   ├── run.sh
│   ├── requirements.txt
│   └── README.md
├── task2_detection/               # 任务2: COCO目标检测
│   ├── train.py
│   ├── evaluate.py
│   ├── run.sh
│   ├── requirements.txt
│   └── README.md
└── task3_segmentation/            # 任务3: Cityscapes语义分割
    ├── train.py
    ├── evaluate.py
    ├── run.sh
    ├── requirements.txt
    └── README.md
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/TomatoHY/DeepLearning_HW2.git
cd DeepLearning_HW2
```

### 2. 下载数据集和模型

**重要**: 由于数据集和模型文件较大，未包含在 Git 仓库中。请使用以下方式下载：

#### 方式一：使用下载脚本（推荐）

```bash
bash download.sh
```

该脚本会自动下载：
- CIFAR-10 数据集（通过 torchvision 自动下载）
- COCO 2017 数据集（train/val）
- Stanford Background 数据集
- 预训练模型权重

#### 方式二：手动下载

如果自动下载失败，可以手动下载数据集：

**CIFAR-10**:
- 训练时会自动下载到 `./datasets/cifar10/`

**COCO 2017**:
```bash
mkdir -p datasets/coco
cd datasets/coco
# 下载训练集图像 (18GB)
wget http://images.cocodataset.org/zips/train2017.zip
# 下载验证集图像 (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
# 下载标注文件
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# 解压
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

**Stanford Background**:
```bash
mkdir -p datasets/stanford_background
cd datasets/stanford_background
# 下载数据集
wget http://dags.stanford.edu/data/iccv09Data.tar.gz
tar -xzf iccv09Data.tar.gz
```

**注意**: 
- 数据集总大小约 20GB，请确保有足够的磁盘空间
- 下载可能需要较长时间，建议使用稳定的网络连接
- 模型 checkpoints 需要训练后生成，或从其他来源获取

### 3. 安装依赖

每个任务目录下都有独立的 `requirements.txt`：

```bash
# Task 1
cd task1_classification
pip install -r requirements.txt

# Task 2
cd task2_detection
pip install -r requirements.txt

# Task 3
cd task3_segmentation
pip install -r requirements.txt
```

### 4. 运行训练

每个任务都提供了 `run.sh` 脚本：

```bash
# Task 1: CIFAR-10分类
cd task1_classification
bash run.sh

# Task 2: COCO目标检测
cd task2_detection
bash run.sh

# Task 3: Cityscapes分割
cd task3_segmentation
bash run.sh
```

## 任务说明

### Task 1: CIFAR-10 图像分类
- **模型**: ResNet18
- **数据集**: CIFAR-10 (10类，60K图像)
- **最佳准确率**: 82.13%
- **训练时间**: ~2-3小时 (单GPU)

### Task 2: PASCAL VOC 目标检测
- **模型**: Faster R-CNN (ResNet-50-FPN)
- **数据集**: COCO 2017 (80类)
- **检测准确率**: 19.52% / 精确率: 94.95%
- **训练时间**: ~8-10小时 (单GPU)

### Task 3: Stanford Background 语义分割
- **模型**: U-Net (从零实现)
- **数据集**: Stanford Background (8类)
- **最佳 mIoU**: 56.94% (Dice Loss)
- **训练时间**: ~4-6小时 (单GPU)

## 服务器路径配置

所有脚本已配置为使用相对路径：
- 数据集: `./datasets/`
- 模型: `./models/`

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7 (推荐)
- GPU内存: 至少8GB

## 常见问题

1. **CUDA out of memory**: 减小 batch_size
2. **数据加载慢**: 增加 num_workers
3. **数据集下载失败**: 使用手动下载方式，或检查网络连接
4. **checkpoint 文件过大**: 已在 `.gitignore` 中排除，不会上传到 Git

## 文件说明

- **数据集和模型**: 由于文件过大，未包含在仓库中，请使用 `download.sh` 下载
- **训练结果**: checkpoints、可视化结果等文件已被 `.gitignore` 排除
- **实验报告**: 完整的实验报告请查看 `complete_report.md`

## 详细文档

每个任务的详细说明请查看对应目录下的 `README.md` 文件。
