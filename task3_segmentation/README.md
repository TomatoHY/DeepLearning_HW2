# Task 3: 从零搭建 U-Net 进行图像分割

## 任务要求

1. **不使用任何预训练权重**，从零开始手写搭建 U-Net
2. 使用 **Stanford Background Dataset** 数据集
3. 手动实现 **Dice Loss**
4. 对比三种损失函数配置：
   - Cross-Entropy Loss
   - Dice Loss
   - Combined Loss (CE + Dice)

## 项目结构

```
task3_segmentation/
├── train_unet.py           # 训练脚本（包含 U-Net 实现和 Dice Loss）
├── evaluate_unet.py        # 评估脚本
├── run_task3.sh           # 一键运行脚本
├── requirements.txt       # 依赖包
└── README_TASK3.md        # 本文档
```

## 网络结构

### U-Net 架构

完整的 U-Net 实现，包含：

1. **编码器（下采样路径）**：
   - 初始卷积块：3 → 64 channels
   - Down1: 64 → 128 channels
   - Down2: 128 → 256 channels
   - Down3: 256 → 512 channels
   - Down4: 512 → 1024 channels（瓶颈层）

2. **解码器（上采样路径）**：
   - Up1: 1024 → 512 channels
   - Up2: 512 → 256 channels
   - Up3: 256 → 128 channels
   - Up4: 128 → 64 channels

3. **Skip Connections**：
   - 每个上采样层都与对应的下采样层特征拼接
   - 保留空间信息，提升分割精度

4. **输出层**：
   - 1×1 卷积：64 → 8 channels（8个类别）

### 关键组件

- **DoubleConv**: 两个 3×3 卷积 + BatchNorm + ReLU
- **Down**: MaxPool2d + DoubleConv
- **Up**: ConvTranspose2d + Concat + DoubleConv

## 损失函数

### 1. Cross-Entropy Loss
标准的交叉熵损失，适用于多类别分割任务。

### 2. Dice Loss（手动实现）
```python
Dice Coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
Dice Loss = 1 - Dice Coefficient
```

优点：
- 对类别不平衡问题更鲁棒
- 直接优化 IoU 相关指标

### 3. Combined Loss
```python
Combined Loss = 0.5 * CE Loss + 0.5 * Dice Loss
```

结合两者优势：
- CE Loss 提供稳定的梯度
- Dice Loss 处理类别不平衡

## 数据集

### Stanford Background Dataset

- **类别数**: 8
  - sky（天空）
  - tree（树木）
  - road（道路）
  - grass（草地）
  - water（水域）
  - building（建筑）
  - mountain（山脉）
  - foreground object（前景物体）

- **图像数量**: ~700 张
- **分辨率**: 320×240（训练时 resize 到 256×256）

## 环境配置

### 依赖安装

```bash
pip install torch torchvision numpy pillow matplotlib tqdm
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

### 数据集准备

1. 下载 Stanford Background Dataset
2. 解压到指定目录，结构如下：

```
stanford_background/
├── images/
│   ├── 0000001.jpg
│   ├── 0000002.jpg
│   └── ...
├── labels/
│   ├── 0000001.regions.png
│   ├── 0000002.regions.png
│   └── ...
├── train.txt
└── test.txt
```

## 训练

### 方法 1: 一键运行（推荐）

训练所有三种损失函数配置：

```bash
chmod +x run_task3.sh
./run_task3.sh
```

### 方法 2: 单独训练

#### Cross-Entropy Loss
```bash
python train_unet.py \
    --data_path /path/to/stanford_background \
    --loss_type ce \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
```

#### Dice Loss
```bash
python train_unet.py \
    --data_path /path/to/stanford_background \
    --loss_type dice \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
```

#### Combined Loss
```bash
python train_unet.py \
    --data_path /path/to/stanford_background \
    --loss_type combined \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | - | 数据集路径 |
| `--save_dir` | `./checkpoints` | 模型保存目录 |
| `--loss_type` | `ce` | 损失函数类型：ce/dice/combined |
| `--batch_size` | 8 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--img_size` | 256 | 图像大小 |
| `--num_workers` | 4 | 数据加载线程数 |

## 评估

```bash
python evaluate.py \
    --data_path /mnt/data/kw/hy/projects/course/DL/hw2/datasets/stanford_background \
    --checkpoint ./checkpoints/best_model_dice.pth \
    --visualize \
    --num_vis 10
```

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | - | 模型权重路径 |
| `--visualize` | False | 是否生成可视化 |
| `--num_vis` | 10 | 可视化图像数量 |
| `--save_dir` | `./visualizations` | 可视化保存目录 |

## 输出文件

### 训练输出

```
checkpoints/
├── best_model_ce.pth          # CE Loss 最佳模型
├── best_model_dice.pth        # Dice Loss 最佳模型
├── best_model_combined.pth    # Combined Loss 最佳模型
├── history_ce.json            # CE Loss 训练历史
├── history_dice.json          # Dice Loss 训练历史
└── history_combined.json      # Combined Loss 训练历史
```

### 评估输出

```
visualizations_ce/
├── segmentation_0.png
├── segmentation_1.png
└── ...

visualizations_dice/
├── segmentation_0.png
└── ...

visualizations_combined/
├── segmentation_0.png
└── ...
```

每张可视化图包含：
- 输入图像
- Ground Truth 分割
- 模型预测分割

## 实验结果对比

训练完成后，可以对比三种损失函数的性能：

| Loss Type | Val mIoU | 说明 |
|-----------|----------|------|
| Cross-Entropy | - | 标准损失函数 |
| Dice Loss | - | 对类别不平衡更鲁棒 |
| Combined | - | 结合两者优势 |

## 关键实现细节

### 1. U-Net 实现

- **编码器**: 4层下采样，每层特征通道数翻倍
- **解码器**: 4层上采样，使用转置卷积
- **Skip Connections**: 通过 `torch.cat` 拼接特征
- **尺寸处理**: 自动处理上采样后的尺寸不匹配问题

### 2. Dice Loss 实现

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 将 logits 转换为概率
        pred = F.softmax(pred, dim=1)
        
        # 将 target 转换为 one-hot 编码
        target_one_hot = F.one_hot(target, num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # 计算每个类别的 Dice coefficient
        intersection = (pred * target_one_hot).sum()
        union = pred.sum() + target_one_hot.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice
```

### 3. 数据增强

- Resize 到 256×256
- ImageNet 标准化
- 标签使用最近邻插值（保持类别标签不变）

### 4. 训练策略

- **优化器**: Adam (lr=1e-4)
- **学习率调度**: ReduceLROnPlateau (patience=5, factor=0.5)
- **早停**: 保存验证集 mIoU 最高的模型

## 注意事项

1. **不使用预训练权重**: 所有模型从随机初始化开始训练
2. **GPU 内存**: batch_size=8 需要约 4GB GPU 内存
3. **训练时间**: 每个配置约需 1-2 小时（取决于 GPU）
4. **数据集路径**: 确保修改为实际的数据集路径

## 服务器运行

修改 `run_task3.sh` 中的数据路径：

```bash
DATA_PATH="/mnt/data/kw/hy/projects/course/DL/hw2/datasets/stanford_background"
```

然后运行：

```bash
chmod +x run_task3.sh
./run_task3.sh
```

## 参考资料

- U-Net 论文: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Dice Loss 论文: [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
- Stanford Background Dataset: [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html)
