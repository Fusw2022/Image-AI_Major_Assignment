import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 使像素值保持在 [0, 1] 范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接中的捷径(shortcut)部分
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

class RCNN(nn.Module):
    def __init__(self, num_classes,img_size):
        super(RCNN, self).__init__()
        # 使用预训练的 ResNet 作为特征提取器
        self.feature_extractor = models.resnet18(pretrained=True)
        # 移除最后的全连接层
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        # 边界框回归器
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)  # 输出 4 个坐标值 (x, y, w, h)
        )

    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        # 分类
        class_scores = self.classifier(features)
        # 边界框回归
        bbox_preds = self.bbox_regressor(features)
        return class_scores, bbox_preds

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super(EnhancedCNN, self).__init__()
        self.in_channels = 32

        # 初始卷积层（第1层）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # 第1次池化：尺寸减半

        # 残差块（第2-5层，共4个残差块）
        self.res_block1 = ResidualBlock(32, 64, stride=1)   # 第2层
        self.res_block2 = ResidualBlock(64, 128, stride=1)  # 第3层
        self.res_block3 = ResidualBlock(128, 256, stride=1) # 第4层
        self.res_block4 = ResidualBlock(256, 512, stride=1) # 第5层（新增第4个残差块）

        # 计算全连接层输入维度（考虑池化次数）
        # 初始池化1次，残差块后各池化1次（共4次池化）
        self.fc_input_dim = img_size * img_size * 2  # 通道数为残差块4的输出512

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 第1次池化：尺寸/2

        # 残差块+池化（每次残差块后池化，共4次池化）
        x = self.pool(self.res_block1(x))  # 第2次池化：尺寸/4
        x = self.pool(self.res_block2(x))  # 第3次池化：尺寸/8
        x = self.pool(self.res_block3(x))  # 第4次池化：尺寸/16
        x = self.res_block4(x)  # 第4个残差块（不池化，仅卷积）

        # 展平并通过全连接层
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

class CNNModel2(nn.Module):#EnhancedCNN
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

class CNNModel3(nn.Module):#VGGNet
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

class EnhancedCNN2_4(nn.Module):
    def __init__(self, num_classes, img_size):
        super(EnhancedCNN2_4, self).__init__()
        self.in_channels = 32

        # 初始卷积层（第1层）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # 第1次池化：尺寸减半

        # 新增的卷积层
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 新增的残差块
        self.res_block_extra1 = ResidualBlock(32, 32, stride=1)
        self.res_block_extra2 = ResidualBlock(32, 32, stride=1)
        self.res_block_extra3 = ResidualBlock(32, 32, stride=1)

        # 原有的残差块（第2-5层，共4个残差块）
        self.res_block1 = ResidualBlock(32, 64, stride=1)   # 第2层
        self.res_block2 = ResidualBlock(64, 128, stride=1)  # 第3层
        self.res_block3 = ResidualBlock(128, 256, stride=1) # 第4层
        self.res_block4 = ResidualBlock(256, 512, stride=1) # 第5层（新增第4个残差块）

        # 计算全连接层输入维度（考虑池化次数）
        # 初始池化1次，新增卷积层后池化1次，残差块后各池化1次（共5次池化）
        # 尺寸变化：img_size → img_size/2 → img_size/4 → img_size/8 → img_size/16 → img_size/32
        feature_size = img_size // (2 ** 5)  # 2^5=32，最终尺寸为 img_size/32
        self.fc_input_dim = feature_size * feature_size * 512  # 通道数为残差块4的输出512

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 第1次池化：尺寸/2
        x = F.relu(self.bn2(self.conv2(x)))

        # 通过新增的残差块
        x = self.res_block_extra1(x)
        x = self.res_block_extra2(x)
        x = self.res_block_extra3(x)

        x = self.pool(x)  # 新增卷积层后池化：尺寸/4

        # 残差块+池化（每次残差块后池化，共4次池化）
        x = self.pool(self.res_block1(x))  # 第3次池化：尺寸/8
        x = self.pool(self.res_block2(x))  # 第4次池化：尺寸/16
        x = self.pool(self.res_block3(x))  # 第5次池化：尺寸/32
        x = self.res_block4(x)  # 第4个残差块（不池化，仅卷积）

        # 展平并通过全连接层
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EnhancedCNN6(nn.Module):
    def __init__(self, num_classes, img_size):
        super(EnhancedCNN6, self).__init__()
        self.in_channels = 32

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # 使用残差块替代部分普通卷积层 (扩展到6层)
        self.res_block1 = ResidualBlock(32, 64, stride=1)
        self.res_block2 = ResidualBlock(64, 128, stride=1)
        self.res_block3 = ResidualBlock(128, 256, stride=1)
        self.res_block4 = ResidualBlock(256, 512, stride=1)
        self.res_block5 = ResidualBlock(512, 768, stride=1)
        self.res_block6 = ResidualBlock(768, 1024, stride=1)

        # 计算全连接层输入维度 (修正后的计算)
        self.fc_input_dim = img_size * img_size * 4

        # 全连接层分类器
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 第1次池化

        # 应用残差块和池化
        x = self.pool(self.res_block1(x))  # 第2次池化
        x = self.pool(self.res_block2(x))  # 第3次池化
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.pool(self.res_block5(x))  # 第4次池化
        x = self.res_block6(x)

        # 展平并通过全连接层
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EnhancedCNN3(nn.Module):
    def __init__(self, num_classes,img_size):
        super(EnhancedCNN3, self).__init__()
        self.in_channels = 32

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # 使用残差块替代部分普通卷积层
        self.res_block1 = ResidualBlock(32, 64, stride=1)
        self.res_block2 = ResidualBlock(64, 128, stride=1)
        self.res_block3 = ResidualBlock(128, 256, stride=1)

        # 计算全连接层输入维度
        feature_size = img_size
        self.fc_input_dim = feature_size * feature_size

        # 全连接层分类器
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 应用残差块
        x = self.pool(self.res_block1(x))
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))

        # 展平并通过全连接层
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EnhancedMediumCNN(nn.Module):
    """改进版的 MediumCNN，集成了抗过拟合策略"""
    def __init__(self, num_classes, img_size=224, use_bn=True, dropout_rate=0.5):
        super(EnhancedMediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.3)  # 增大空间 Dropout

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)  # 增大空间 Dropout

        # 使用全局平均池化替代全连接层，减少参数
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout3 = nn.Dropout(dropout_rate)  # 全连接层前的 Dropout
        self.fc = nn.Linear(64, num_classes)
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
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout3(x)
        x = self.fc(x)
        return x