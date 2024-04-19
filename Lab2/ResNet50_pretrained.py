import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50_pretrained(nn.Module):
    def __init__(self, num_class=100):
        super(ResNet50_pretrained, self).__init__()
        self.ResNet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.ResNet50.fc = nn.Linear(2048, num_class)
        
    def forward(self, x):
        x = self.ResNet50(x)
        return x