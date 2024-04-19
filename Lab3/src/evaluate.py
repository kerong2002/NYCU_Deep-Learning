import numpy as np
import torch
import torch.nn as nn
    
from utils import dice_loss, dice_score
    
def evaluate(net, data, device):
    val_loss = []
    val_dice_score = []
    criterion = nn.BCELoss()
    with torch.no_grad():
        net.eval()
        for batch in data:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            pred_mask = net(image)
            val_loss.append(criterion(pred_mask, mask).item() + dice_loss(pred_mask, mask).item())
            val_dice_score.append(dice_score(pred_mask, mask).item())
        print(f"val losses: {np.mean(val_loss)}, val dice score: {np.mean(val_dice_score)}")
    return val_loss, val_dice_score
        