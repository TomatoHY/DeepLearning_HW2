import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import argparse
from tqdm import tqdm
import json
from datasets import load_dataset
from PIL import Image
import numpy as np

def get_model(num_classes):
    # 加载预训练的Faster R-CNN模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

class HFCOCODataset(Dataset):
    """使用Hugging Face格式的COCO数据集"""
    def __init__(self, hf_dataset, transforms=None):
        self.dataset = hf_dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 获取图像
        image = item['image']

        # 处理不同的图像格式
        if isinstance(image, dict):
            # 如果是字典格式，尝试获取bytes或path
            if 'bytes' in image:
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image['bytes']))
            elif 'path' in image:
                from PIL import Image
                image = Image.open(image['path'])
            else:
                raise ValueError(f"Unknown image format: {image.keys()}")

        # 确保是RGB格式
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')

        # 转换为tensor
        image = torchvision.transforms.functional.to_tensor(image)

        # 获取标注
        objects = item['objects']
        boxes = []
        labels = []
        areas = []

        for i in range(len(objects['bbox'])):
            bbox = objects['bbox'][i]
            # COCO格式: [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(objects['category'][i])  # 使用 'category' 而不是 'category_id'
            areas.append(objects['area'][i])

        # 转换为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        image_id = torch.tensor([item['image_id']])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.dataset)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0.0

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for i, (images, targets) in enumerate(pbar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix({'loss': avg_loss})

    return running_loss / len(data_loader)

@torch.no_grad()
def evaluate_simple(model, data_loader, device):
    """简单评估：计算平均精度"""
    model.eval()
    total_correct = 0
    total_objects = 0

    for images, targets in tqdm(data_loader, desc='Evaluating'):
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for target, output in zip(targets, outputs):
            # 简单统计：检测到的目标数量
            total_objects += len(target['boxes'])
            # 统计高置信度检测
            high_conf = (output['scores'] > 0.5).sum().item()
            total_correct += min(high_conf, len(target['boxes']))

    accuracy = total_correct / max(total_objects, 1) * 100
    return accuracy

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载Hugging Face格式的COCO数据集
    print('Loading COCO dataset from Hugging Face format...')

    # 直接从data目录加载parquet文件
    data_dir = os.path.join(args.data_path, 'data')

    if os.path.exists(data_dir):
        print(f'Loading from parquet files in: {data_dir}')
        # 使用load_dataset加载本地parquet文件
        dataset = load_dataset('parquet', data_dir=data_dir, split='train')

        # 分割训练集和验证集
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset_hf = dataset_split['train']
        val_dataset_hf = dataset_split['test']
    else:
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 包装为PyTorch Dataset
    train_dataset = HFCOCODataset(train_dataset_hf)
    val_dataset = HFCOCODataset(val_dataset_hf)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # 模型 (COCO有80个类别 + 背景)
    num_classes = 91
    model = get_model(num_classes).to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    # 学习率调度
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch+1, args.print_freq)
        lr_scheduler.step()

        history['train_loss'].append(train_loss)
        print(f'Train Loss: {train_loss:.4f}')

        # 每3个epoch评估一次
        if (epoch + 1) % 3 == 0:
            print('Running evaluation...')
            val_acc = evaluate_simple(model, val_loader, device)
            history['val_acc'].append(val_acc)
            print(f'Validation Accuracy: {val_acc:.2f}%')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f'Best model saved with accuracy: {best_acc:.2f}%')

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.save_dir, 'latest_model.pth'))

    # 保存训练历史
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COCO Object Detection with Faster R-CNN')
    parser.add_argument('--data_path', type=str, default='/mnt/data/kw/hy/projects/course/DL/hw2/datasets/coco',
                       help='Path to COCO dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=12,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--print_freq', type=int, default=100,
                       help='Print frequency')

    args = parser.parse_args()
    main(args)
