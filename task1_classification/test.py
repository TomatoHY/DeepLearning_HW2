import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_model(num_classes=10):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Confusion matrix saved to {save_path}')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)

    # 加载模型
    model = get_model(num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint}')
    print(f'Best training accuracy: {checkpoint["best_acc"]:.2f}%')

    # 测试
    predictions, labels = test(model, test_loader, device)

    # 计算准确率
    accuracy = 100 * np.sum(predictions == labels) / len(labels)
    print(f'\nTest Accuracy: {accuracy:.2f}%')

    # 分类报告
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    print('\nClassification Report:')
    print(classification_report(labels, predictions, target_names=classes))

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, classes, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CIFAR-10 Classification Model')
    parser.add_argument('--data_path', type=str, default='/mnt/data/kw/hy/projects/course/DL/hw2/datasets/cifar10',
                       help='Path to CIFAR-10 dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_path', type=str, default='./confusion_matrix.png',
                       help='Path to save confusion matrix')

    args = parser.parse_args()
    main(args)
