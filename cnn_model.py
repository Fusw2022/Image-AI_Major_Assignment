import torch
import torch.nn as nn
import torch.nn.functional as F

class MediumCNN(nn.Module):
    """中等复杂度的CNN模型，包含批标准化和Dropout"""

    def __init__(self, num_classes, img_size=224, use_bn=True):
        super(MediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        # 修正全连接层输入维度，适应224×224输入
        self.fc1 = nn.Linear(64 * (img_size//4) * (img_size//4), 512)
        self.bn5 = nn.BatchNorm1d(512) if use_bn else nn.Identity()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward_conv(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.flatten(x)
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class EnhancedCNN(nn.Module):
    def __init__(self,num_classes, img_size=224):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 在这里添加一个新的卷积层、BatchNorm和相应的池化层
        # 添加新的卷积层
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 添加 BatchNorm 层
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # 修改全连接层以适应新的特征图尺寸
        self.fc1 = nn.Linear(64 * (img_size//4) * (img_size//4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        # 实现包含新卷积层的前向传播
        # 第一个卷积层及激活函数
        x = self.conv1(x)
        x = self.relu(x)
        # 第一个池化层
        x = self.pool(x)
        # 第二个卷积层及激活函数
        x = self.conv2(x)
        x = self.relu(x)
        # 第二个池化层
        x = self.pool(x)
        # 新增的卷积层、BatchNorm 层及激活函数
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # 展平操作
        x = self.flatten(x)
        # 第一个全连接层及激活函数
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接层
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel1(nn.Module):
    def __init__(self, num_classes,img_size):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 计算并保存全连接层的输入维度
        self.fc_input_dim = img_size*img_size*2
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNModel2(nn.Module):
    def __init__(self, num_classes,img_size):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc_input_dim = img_size * img_size * 2
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNModel3(nn.Module):
    def __init__(self, num_classes,img_size):
        super(CNNModel3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_input_dim = img_size * img_size * 4
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.fc_input_dim)
        x = self.classifier(x)
        return x