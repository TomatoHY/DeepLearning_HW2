"""
Task 3: 从零搭建 U-Net 进行图像分割
要求：
1. 不使用任何预训练权重
2. 手写 U-Net 网络结构
3. 使用 Stanford Background Dataset
4. 实现 Dice Loss
5. 对比三种损失函数配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


class DoubleConv(nn.Module):
    """U-Net 的双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块：MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块：UpConv + Concat + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Skip connection: 拼接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net 网络结构
    包含：
    - 编码器（下采样）：4层
    - 瓶颈层
    - 解码器（上采样）：4层
    - Skip connections
    """
    def __init__(self, n_channels=3, n_classes=8):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 编码器（下采样路径）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 解码器（上采样路径）
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 输出层
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器（带 skip connections）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出
        logits = self.outc(x)
        return logits


class DiceLoss(nn.Module):
    """
    手动实现的 Dice Loss
    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - 模型输出的 logits
        target: (B, H, W) - ground truth labels
        """
        # 创建有效像素的mask（排除标签为-1的像素）
        valid_mask = (target != -1)

        # 将 logits 转换为概率
        pred = F.softmax(pred, dim=1)

        # 将 target 中的 -1 替换为 0（临时处理，后面会用 mask 过滤）
        target_masked = target.clone()
        target_masked[~valid_mask] = 0

        # 将 target 转换为 one-hot 编码
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target_masked, num_classes=num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # 扩展 mask 到 one-hot 维度
        valid_mask_expanded = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)

        # 计算每个类别的 Dice coefficient
        dice_scores = []
        for c in range(num_classes):
            pred_c = pred[:, c, :, :] * valid_mask.float()
            target_c = target_one_hot[:, c, :, :] * valid_mask.float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                dice_scores.append(dice)

        # 平均 Dice coefficient
        if dice_scores:
            dice_coef = torch.stack(dice_scores).mean()
        else:
            dice_coef = torch.tensor(0.0, device=pred.device)

        # Dice Loss = 1 - Dice Coefficient
        return 1 - dice_coef


class CombinedLoss(nn.Module):
    """组合损失：Cross-Entropy + Dice Loss"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class StanfordBackgroundDataset(Dataset):
    """Stanford Background Dataset 数据加载器"""

    @staticmethod
    def parse_regions_txt(txt_path, img_width, img_height):
        """
        解析 .regions.txt 文件生成标签图像

        文件格式：每行包含一行像素的标签，用空格分隔
        从上到下，每行从左到右
        标签值 -1 表示未标注区域
        """
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # 解析每行的标签值
        labels = []
        for line in lines:
            row_labels = [int(x) for x in line.split()]
            labels.extend(row_labels)

        # 转换为图像（保持为 int32 以支持 -1 值）
        label_array = np.array(labels, dtype=np.int32).reshape(img_height, img_width)
        return Image.fromarray(label_array, mode='I')

    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        root_dir: 数据集根目录
        split: 'train' 或 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # 图像和标签目录
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')

        # 检查目录是否存在
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            print(f"Warning: Labels directory not found: {self.labels_dir}")

        # 读取文件列表
        split_file = os.path.join(root_dir, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                # 读取文件名并构建完整路径
                filenames = [line.strip() for line in f.readlines() if line.strip()]
                self.image_files = [os.path.join(self.images_dir, fname) if not os.path.isabs(fname) else fname
                                   for fname in filenames]
            print(f"Loaded {len(self.image_files)} images from {split_file}")
        else:
            # 如果没有 split 文件，使用所有图像
            self.image_files = sorted([os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
                                      if f.endswith(('.jpg', '.png'))])
            print(f"No split file found, using all {len(self.image_files)} images from {self.images_dir}")

            # 如果还是没有图像，尝试递归查找
            if len(self.image_files) == 0:
                print(f"No images found in {self.images_dir}, searching recursively...")
                for root, dirs, files in os.walk(root_dir):
                    for f in files:
                        if f.endswith(('.jpg', '.png')) and 'region' not in f.lower():
                            self.image_files.append(os.path.join(root, f))
                print(f"Found {len(self.image_files)} images recursively")

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {root_dir}. Please check the dataset structure.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像（现在 image_files 已经是完整路径）
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)

        image = Image.open(img_path).convert('RGB')

        # 加载标签
        # Stanford Background 标签格式: xxx.regions.txt
        label_name = base_name.replace('.jpg', '.regions.txt')
        label_path = os.path.join(self.labels_dir, label_name)

        if os.path.exists(label_path):
            # 解析 .regions.txt 文件
            label = self.parse_regions_txt(label_path, image.width, image.height)
        else:
            # 如果标签不存在，创建空标签
            print(f"Warning: Label not found for {base_name}, using empty label")
            label = Image.new('L', image.size, 0)

        # 应用变换
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_transforms(img_size=256):
    """获取数据变换"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.long().squeeze(0))
    ])

    return transform, target_transform


def calculate_miou(pred, target, num_classes):
    """计算 mIoU（忽略标签为-1的像素）"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # 创建有效像素的mask（排除标签为-1的像素）
    valid_mask = (target != -1)

    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid_mask
        target_cls = (target == cls) & valid_mask

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            iou = intersection / union
            ious.append(iou.item())

    return np.mean(ious) if ious else 0.0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    running_miou = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算 mIoU
        preds = outputs.argmax(dim=1)
        miou = calculate_miou(preds, labels, num_classes=8)
        running_miou += miou

        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'mIoU': running_miou / (pbar.n + 1)
        })

    return running_loss / len(train_loader), running_miou / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    running_miou = 0.0

    for images, labels in tqdm(val_loader, desc='Validation'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # 计算 mIoU
        preds = outputs.argmax(dim=1)
        miou = calculate_miou(preds, labels, num_classes=8)
        running_miou += miou

    return running_loss / len(val_loader), running_miou / len(val_loader)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loss type: {args.loss_type}')

    # 数据加载
    transform, target_transform = get_transforms(args.img_size)

    train_dataset = StanfordBackgroundDataset(
        root_dir=args.data_path,
        split='train',
        transform=transform,
        target_transform=target_transform
    )

    val_dataset = StanfordBackgroundDataset(
        root_dir=args.data_path,
        split='test',
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    # 模型（从零开始，不使用预训练）
    model = UNet(n_channels=3, n_classes=8).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # 损失函数（忽略标签值为-1的像素）
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.loss_type == 'dice':
        criterion = DiceLoss()
    elif args.loss_type == 'combined':
        criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    else:
        raise ValueError(f'Unknown loss type: {args.loss_type}')

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # 训练
    os.makedirs(args.save_dir, exist_ok=True)
    best_miou = 0
    history = {
        'train_loss': [], 'train_miou': [],
        'val_loss': [], 'val_miou': []
    }

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')

        train_loss, train_miou = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou = validate(model, val_loader, criterion, device)

        scheduler.step(val_miou)

        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_miou)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)

        print(f'Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}')

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'loss_type': args.loss_type
            }, os.path.join(args.save_dir, f'best_model_{args.loss_type}.pth'))
            print(f'✓ Best model saved with mIoU: {best_miou:.4f}')

    # 保存训练历史
    with open(os.path.join(args.save_dir, f'history_{args.loss_type}.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print(f'\nTraining completed! Best mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-Net for Stanford Background Dataset')
    parser.add_argument('--data_path', type=str,
                       default='/mnt/data/kw/hy/projects/course/DL/hw2/datasets/stanford_background',
                       help='Path to Stanford Background dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--loss_type', type=str, default='ce',
                       choices=['ce', 'dice', 'combined'],
                       help='Loss function: ce (Cross-Entropy), dice (Dice Loss), combined (CE+Dice)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()
    main(args)
