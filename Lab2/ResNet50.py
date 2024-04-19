import torch
import torch.nn as nn
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, down=False):
        super(BottleneckBlock, self).__init__()
        if in_channels != out_channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=2 if down else 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * expansion)
            )
        else:
            self.shortcut = nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2 if down else 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * expansion)
        )
        self.ReLU = nn.ReLU()
    def forward(self, x):
        return self.ReLU(self.block(x) + self.shortcut(x))

class ResNet50(nn.Module):
    def __init__(self, num_class=100):
        super(ResNet50, self).__init__()
        self.expansion = 4
        self.channels = [64, 128, 256, 512]
        self.num_blocks = [3, 4, 6, 3]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.build_layer(self.channels[0], self.channels[0] , self.num_blocks[0], False)
        self.conv3 = self.build_layer(self.channels[0] * self.expansion, self.channels[1], self.num_blocks[1], True)
        self.conv4 = self.build_layer(self.channels[1] * self.expansion, self.channels[2], self.num_blocks[2], True)
        self.conv5 = self.build_layer(self.channels[2] * self.expansion, self.channels[3], self.num_blocks[3], True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.channels[3] * self.expansion, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def build_layer(self, in_channels, out_channels, num_blocks, down):
        layer = [BottleneckBlock(in_channels, out_channels, self.expansion, down)]
        for _ in range(1, num_blocks):
            layer.append(BottleneckBlock(out_channels * self.expansion, out_channels, self.expansion))
        return nn.Sequential(*layer)
              
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    
if __name__ == "__main__":
    x = torch.randn(32, 3, 224, 224)
    model = ResNet50()
    print(model(x).shape)