#!/usr/bin/env python3
"""
快速测试 U-Net 实现是否正确
"""

import torch
from train import UNet, DiceLoss, CombinedLoss

def test_unet():
    """测试 U-Net 前向传播"""
    print("Testing U-Net...")

    model = UNet(n_channels=3, n_classes=8)
    x = torch.randn(2, 3, 256, 256)  # batch_size=2, 3 channels, 256x256

    output = model(x)

    assert output.shape == (2, 8, 256, 256), f"Expected (2, 8, 256, 256), got {output.shape}"
    print(f"✓ U-Net output shape: {output.shape}")

    # 计算参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {params/1e6:.2f}M")


def test_dice_loss():
    """测试 Dice Loss"""
    print("\nTesting Dice Loss...")

    criterion = DiceLoss()

    # 模拟输出和标签
    pred = torch.randn(2, 8, 256, 256)  # logits
    target = torch.randint(0, 8, (2, 256, 256))  # labels

    loss = criterion(pred, target)

    assert loss.item() >= 0 and loss.item() <= 1, f"Dice loss should be in [0, 1], got {loss.item()}"
    print(f"✓ Dice Loss: {loss.item():.4f}")


def test_combined_loss():
    """测试组合损失"""
    print("\nTesting Combined Loss...")

    criterion = CombinedLoss()

    pred = torch.randn(2, 8, 256, 256)
    target = torch.randint(0, 8, (2, 256, 256))

    loss = criterion(pred, target)

    print(f"✓ Combined Loss: {loss.item():.4f}")


def test_backward():
    """测试反向传播"""
    print("\nTesting backward pass...")

    model = UNet(n_channels=3, n_classes=8)
    criterion = DiceLoss()

    x = torch.randn(2, 3, 256, 256)
    target = torch.randint(0, 8, (2, 256, 256))

    output = model(x)
    loss = criterion(output, target)
    loss.backward()

    # 检查梯度
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    print("✓ Backward pass successful")


if __name__ == '__main__':
    print("="*50)
    print("U-Net Implementation Test")
    print("="*50)

    test_unet()
    test_dice_loss()
    test_combined_loss()
    test_backward()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
