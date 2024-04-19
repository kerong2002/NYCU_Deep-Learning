import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from utils import dice_score, focal_loss, dice_loss
from evaluate import evaluate

def train(args):
    train_data = load_dataset(args.data_path, mode="train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data = load_dataset(args.data_path, mode="valid")
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    if args.model == "unet":
        model = UNet(3, 1).to(args.device)
    else:
        model = ResNet34_UNet(3, 1).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    writer = SummaryWriter(f"runs/{args.model}/")
    best_dice_score = 0.88
    
    for epoch in range(args.epochs):
        train_loss = []
        train_dice_score = []
        model.train()
        progress = tqdm(enumerate(train_loader))
        for i, batch in progress:
            image = batch["image"].to(args.device)
            mask = batch["mask"].to(args.device)
            pred_mask = model(image)
            loss = criterion(pred_mask, mask) + dice_loss(pred_mask, mask)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_dice_score.append(dice_score(pred_mask, mask).item())
            progress.set_description((f"Epoch: {epoch + 1}/{args.epochs}, iter: {i + 1}/{len(train_loader)}, Loss: {np.mean(train_loss):.4f}, Dice Score: {np.mean(train_dice_score):.4f}"))
        val_loss, val_dice_score = evaluate(model, val_loader, args.device)
        
        writer.add_scalars(f"Loss", {"train": np.mean(train_loss), "valid": np.mean(val_loss)}, epoch)
        writer.add_scalars(f"Dice Score", {"train": np.mean(train_dice_score), "valid": np.mean(val_dice_score)}, epoch)
        if np.mean(val_dice_score) > best_dice_score:
            best_dice_score = np.mean(val_dice_score)
            torch.save(model, f"../saved_models/{args.model}.pth")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', default="unet", type=str, choices=["unet" ,"resnet34_unet"])
    parser.add_argument('--device', default="cuda", type=str, help='device to use for training')
    parser.add_argument('--data_path', default="../dataset/oxford-iiit-pet/", type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=400, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)