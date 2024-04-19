import torch
import torch.nn as nn
from torchvision.models import vgg19_bn, VGG19_BN_Weights

class VGG19_pretrained(nn.Module):
    def __init__(self, num_class=100):
        super(VGG19_pretrained, self).__init__()
        self.vgg = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
        self.vgg.classifier[-1] = nn.Linear(4096, num_class)
        
    def forward(self, x):
        x = self.vgg(x)
        return x