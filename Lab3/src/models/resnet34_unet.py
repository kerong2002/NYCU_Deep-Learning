import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down = False, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding= 1, 
                               stride= 2 if down else 1, padding_mode="reflect", **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.act(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down = False):
        super(ResidualBlock, self).__init__()
        if down:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, down=down),
            ConvBlock(out_channels, out_channels, down=False),
        )
    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512], num_block=[3, 4, 6, 3]):
        super(ResNet34_UNet, self).__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downs = nn.ModuleList()        
        self.downs.append(self.build_layer(features[0], features[0], num_block[0], False))
        self.downs.append(self.build_layer(features[0], features[1], num_block[0], True))
        self.downs.append(self.build_layer(features[1], features[2], num_block[0], True))
        self.downs.append(self.build_layer(features[2], features[3], num_block[0], True))
        
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2, True)
        
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
                                
        self.last = nn.Sequential(
            nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def build_layer(self, in_channels, out_channels, num_block, down):
        layer = [ResidualBlock(in_channels, out_channels, down)]
        for _ in range(1, num_block):
            layer.append(ResidualBlock(out_channels, out_channels, False))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.init(x)
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up in self.ups:
            if isinstance(up, nn.ConvTranspose2d):
                x = up(x)
                x = torch.cat((x, skips.pop()), dim=1)
            else:
                x = up(x)
        x = self.last(x)
        return x