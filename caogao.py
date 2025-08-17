import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # 如果输入输出维度不一致，用downsample调整
        self.downsample = None
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out


class FCResNet_Block(nn.Module):
    def __init__(self, input_dim, num_classes, layers=[2,2,2,2]):
        """
        input_dim: 输入特征维度
        num_classes: 输出类别数
        layers: 每层堆叠的 ResidualBlock 数量, 类似 ResNet18 的 [2,2,2,2]
        """
        super(FCResNet_Block, self).__init__()

        # 起始投影层，统一映射到64维
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # 残差层堆叠
        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1])
        self.layer3 = self._make_layer(128, 256, layers[2])
        self.layer4 = self._make_layer(256, 512, layers[3])

        # 输出层
        self.fc_out = nn.Linear(512, num_classes)

    def _make_layer(self, in_features, out_features, blocks):
        layers = []
        layers.append(ResidualBlock(in_features, out_features))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc_in(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc_out(out)
        return out


if __name__ == "__main__":
    # 定义一个类似 ResNet18 的结构
    model = FCResNet_Block(input_dim=100, num_classes=10, layers=[2,2,2,2])
    print(model)

    # 测试一下
    x = torch.randn(32, 100)  # batch=32, 输入维度=100
    y = model(x)
    print(y.shape)  # 期望: torch.Size([32, 10])
