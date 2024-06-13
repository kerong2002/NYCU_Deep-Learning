import torch
import torch.nn as nn
from evaluator import evaluation_model

class aux_discriminator(nn.Module, evaluation_model):
    """Discriminator containing the auxiliary classifier."""
    def __init__(self):
        nn.Module.__init__(self)
        evaluation_model.__init__(self)
        self.resnet18.requires_grad_(False)
    
    def forward(self, x):
        out = self.resnet18(x)
        return out
    
    
