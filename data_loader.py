import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import os
from torch.utils.data import random_split


def get_data_loaders(data_dir, batch_size=32, val_split=0.2, img_size=224, augment=True):
    # 数据预处理
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # 新增随机垂直翻转
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增强颜色抖动
            transforms.RandomCrop(img_size, padding=4),  # 新增随机裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载训练数据集
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, '中药数据集'), transform=train_transform)

    # 分割训练集和验证集
    num_samples = len(train_dataset)
    num_val = int(val_split * num_samples)
    num_train = num_samples - num_val

    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])

    # 为验证集应用测试转换
    val_dataset.dataset.transform = test_transform

    # 加载测试数据集
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, '中药数据测试集'), transform=test_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_dataset.dataset.classes