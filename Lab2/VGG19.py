import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)
    
class VGG19(nn.Module):
    def __init__(self, num_class=100):
        super(VGG19, self).__init__()
        self.block1 = self.build_layer(3, 64, 2)
        self.block2 = self.build_layer(64, 128, 2)
        self.block3 = self.build_layer(128, 256, 4)
        self.block4 = self.build_layer(256, 512, 4)
        self.block5 = self.build_layer(512, 512, 4)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def build_layer(self, in_channels, out_channels, num_blocks):
        layer = [ConvBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layer.append(ConvBlock(out_channels, out_channels))
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.flatten(h)
        h = self.fc(h)
        return h