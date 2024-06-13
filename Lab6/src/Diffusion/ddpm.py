import torch
import torch.nn as nn
from diffusers import UNet2DModel

class conditionalDDPM(nn.Module):
    def __init__(self, num_classes=24, dim=512):
        super().__init__()
        channel = dim // 4
        self.ddpm = UNet2DModel(
            sample_size = 64,
            in_channels = 3,
            out_channels = 3,
            layers_per_block = 2,
            block_out_channels = [channel, channel, channel*2, channel*2, channel*4, channel*4], 
            down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"],
            up_block_types=["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"],
            class_embed_type="identity",
        )
        self.class_embedding = nn.Linear(num_classes, dim)
    
    def forward(self, x, t, label):
        class_embed = self.class_embedding(label)
        return self.ddpm(x, t, class_embed).sample
    
if __name__ == "__main__":
    model = conditionalDDPM()
    print(model)
    print(model(torch.randn(1, 3, 64, 64), 10, torch.randint(0, 1, (1, 24), dtype=torch.float)).shape)