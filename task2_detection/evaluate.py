import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
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

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_objects = 0
    total_predictions = 0

    for images, targets in tqdm(data_loader, desc='Evaluating'):
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for target, output in zip(targets, outputs):
            total_objects += len(target['boxes'])
            high_conf = (output['scores'] > 0.5).sum().item()
            total_predictions += high_conf
            total_correct += min(high_conf, len(target['boxes']))

    accuracy = total_correct / max(total_objects, 1) * 100
    precision = total_correct / max(total_predictions, 1) * 100

    return accuracy, precision

def visualize_predictions(model, dataset, device, num_images=5, save_dir='./visualizations'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # COCO类别名称（简化版）
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    for i in range(min(num_images, len(dataset))):
        img, target = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # 转换图像用于显示
        img_np = img.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_np)

        # 绘制预测框
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:  # 只显示置信度>0.5的预测
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                label_name = coco_names[label-1] if 1 <= label <= len(coco_names) else f'class_{label}'
                ax.text(x1, y1-5, f'{label_name}: {score:.2f}',
                       bbox=dict(facecolor='red', alpha=0.5),
                       fontsize=10, color='white')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'), bbox_inches='tight', dpi=150)
        plt.close()

    print(f'Visualizations saved to {save_dir}')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    print('Loading validation dataset...')

    # 直接从data目录加载parquet文件
    data_dir = os.path.join(args.data_path, 'data')

    if os.path.exists(data_dir):
        print(f'Loading from parquet files in: {data_dir}')
        # 使用load_dataset加载本地parquet文件
        dataset = load_dataset('parquet', data_dir=data_dir, split='train')

        # 分割训练集和验证集（使用相同的seed保持一致）
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        val_dataset_hf = dataset_split['test']
    else:
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    val_dataset = HFCOCODataset(val_dataset_hf)
    print(f'Val dataset size: {len(val_dataset)}')

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    # 加载模型
    num_classes = 91
    model = get_model(num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint}')

    # 评估
    print('\nRunning evaluation...')
    accuracy, precision = evaluate(model, val_loader, device)

    print(f'\n{"="*50}')
    print(f'Evaluation Results:')
    print(f'Detection Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}%')
    print(f'{"="*50}')

    # 可视化
    if args.visualize:
        print('\nGenerating visualizations...')
        visualize_predictions(model, val_dataset, device, num_images=args.num_vis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate COCO Object Detection Model')
    parser.add_argument('--data_path', type=str, default='/mnt/data/kw/hy/projects/course/DL/hw2/datasets/coco',
                       help='Path to COCO dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization images')
    parser.add_argument('--num_vis', type=int, default=5,
                       help='Number of images to visualize')

    args = parser.parse_args()
    main(args)
