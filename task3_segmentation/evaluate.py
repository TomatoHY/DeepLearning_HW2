"""
评估 U-Net 模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from train import UNet, StanfordBackgroundDataset, calculate_miou, get_transforms


def visualize_predictions(model, dataset, device, num_images=5, save_dir='./visualizations'):
    """可视化预测结果"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Stanford Background 类别颜色映射
    colors = np.array([
        [128, 0, 0],      # sky
        [0, 128, 0],      # tree
        [128, 128, 0],    # road
        [0, 0, 128],      # grass
        [128, 0, 128],    # water
        [0, 128, 128],    # building
        [128, 128, 128],  # mountain
        [64, 0, 0],       # foreground object
    ]) / 255.0

    class_names = ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground']

    for i in range(min(num_images, len(dataset))):
        img, target = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        # 反归一化图像
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        target_np = target.cpu().numpy()

        # 创建彩色分割图
        pred_color = np.zeros((*pred.shape, 3))
        target_color = np.zeros((*target_np.shape, 3))

        for class_id in range(len(colors)):
            pred_color[pred == class_id] = colors[class_id]
            target_color[target_np == class_id] = colors[class_id]

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(target_color)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pred_color)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'segmentation_{i}.png'), bbox_inches='tight', dpi=150)
        plt.close()

    print(f'Visualizations saved to {save_dir}')


@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=8):
    """评估模型"""
    model.eval()

    # 每个类别的 IoU
    class_ious = {i: [] for i in range(num_classes)}
    overall_miou = 0.0

    class_names = ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground']

    for images, labels in tqdm(val_loader, desc='Evaluating'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        # 计算每个类别的 IoU
        for cls in range(num_classes):
            pred_cls = (preds == cls)
            target_cls = (labels == cls)

            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            if union > 0:
                iou = (intersection / union).item()
                class_ious[cls].append(iou)

    # 计算平均 IoU
    print('\nPer-class IoU:')
    mean_ious = []
    for cls in range(num_classes):
        if class_ious[cls]:
            mean_iou = np.mean(class_ious[cls])
            mean_ious.append(mean_iou)
            print(f'{class_names[cls]:20s}: {mean_iou*100:.2f}%')
        else:
            print(f'{class_names[cls]:20s}: N/A')

    overall_miou = np.mean(mean_ious) if mean_ious else 0.0

    return overall_miou


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载
    transform, target_transform = get_transforms(args.img_size)

    val_dataset = StanfordBackgroundDataset(
        root_dir=args.data_path,
        split='test',
        transform=transform,
        target_transform=target_transform
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # 加载模型
    model = UNet(n_channels=3, n_classes=8).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f'Loaded checkpoint from {args.checkpoint}')
    print(f'Loss type: {checkpoint.get("loss_type", "unknown")}')
    print(f'Best training mIoU: {checkpoint["best_miou"]:.4f}')

    # 评估
    print('\nRunning evaluation...')
    mean_iou = evaluate(model, val_loader, device)

    print(f'\n{"="*50}')
    print(f'Evaluation Results:')
    print(f'Mean IoU: {mean_iou*100:.2f}%')
    print(f'{"="*50}')

    # 可视化
    if args.visualize:
        print('\nGenerating visualizations...')
        visualize_predictions(model, val_dataset, device, num_images=args.num_vis,
                            save_dir=args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate U-Net Model')
    parser.add_argument('--data_path', type=str,
                       default='/mnt/data/kw/hy/projects/course/DL/hw2/datasets/stanford_background',
                       help='Path to Stanford Background dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization images')
    parser.add_argument('--num_vis', type=int, default=10,
                       help='Number of images to visualize')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')

    args = parser.parse_args()
    main(args)
