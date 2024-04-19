import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from VGG19 import VGG19
from ResNet50 import ResNet50
from VGG19_pretrained import VGG19_pretrained
from ResNet50_pretrained import ResNet50_pretrained
from dataloader import ButterflyMothLoader

def evaluate(args, model, val_loader, criterion):
    with torch.no_grad():
        model.eval()
        total_loss, tot_correct, tot_predict = 0, 0, 0
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            tot_predict += y_pred.size(0)
            tot_correct += (y_pred.argmax(1) == y).sum().item()
        return total_loss / len(val_loader), tot_correct / tot_predict
    
def test(args):
    if args.model == "VGG19":
            model = VGG19().to(args.device)
    elif args.model == "ResNet50":
        model = ResNet50().to(args.device)
    elif args.model == "VGG19_pretrained":
        model = VGG19_pretrained().to(args.device)
    elif args.model == "ResNet50_pretrained":
        model = ResNet50_pretrained().to(args.device)
    print(f"Testing on {args.model}")
    test_data = ButterflyMothLoader(root="dataset/", mode="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)
    model.load_state_dict(torch.load(f"checkpoint/{args.model}/best_val_model_{args.model}.pth"))
    model.eval()
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    acc = 0
    top3_acc = 0
    test_losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(args.device), y.to(args.device)
            output = model(x)
            loss = criterion(output, y)
            test_losses.append(loss)
            acc += (output.argmax(1) == y).sum().item()
            top3_acc += (y in (torch.argsort(output, descending=True)[0][:3]))
        acc /= len(test_data)
        top3_acc /= len(test_data)
    print(f"Test Loss: {sum(test_losses)/len(test_losses):.4f}, Test Acc: {acc * 100:.4f}%, Top3 Acc: {top3_acc * 100:.4f}%")
    
def train(args, model, train_loader, optimizer, criterion):
    model.train()
    total_loss, tot_correct, tot_predict = 0, 0, 0
    train_progress = enumerate(train_loader)
    for i, (x, y) in train_progress:
        x, y = x.to(args.device), y.to(args.device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        tot_predict += y_pred.size(0)
        tot_correct += (y_pred.argmax(1) == y).sum().item()
    return total_loss / len(train_loader), tot_correct / tot_predict
            
    
def main(args):
    if args.model == "VGG19":
        model = VGG19().to(args.device)
    elif args.model == "ResNet50":
        model = ResNet50().to(args.device)
    elif args.model == "VGG19_pretrained":
        model = VGG19_pretrained().to(args.device)
    elif args.model == "ResNet50_pretrained":
        model = ResNet50_pretrained().to(args.device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_data = ButterflyMothLoader(root="dataset/", mode="train")
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=10)
    valid_data= ButterflyMothLoader(root="dataset/", mode="valid")
    val_loader = DataLoader(valid_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=10)

    writer = SummaryWriter(f"runs/{args.model}")
    best_val_acc = 0.88
    best_train_acc = 0.88
    progrssbar = tqdm(range(args.epoch))
    for e in progrssbar:
        train_loss, train_acc = train(args, model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(args, model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"checkpoint/{args.model}/best_val_model_{args.model}.pth")
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), f"checkpoint/{args.model}/best_train_model_{args.model}.pth")
        writer.add_scalars(f"Loss", {"train": train_loss, "valid": val_loss}, e)
        writer.add_scalars(f"Acc", {"train": train_acc, "valid": val_acc}, e)
        progrssbar.set_description(f"Epoch: {e + 1}/{args.epoch}, Train Loss: {train_loss:.4}, Val Loss: {val_loss:.4}, Train Acc: {train_acc:.4}, Val Acc: {val_acc:.4}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="VGG19", choices=["VGG19", "ResNet50", "VGG19_pretrained", "ResNet50_pretrained"])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    if args.mode == "train":
        main(args)
    else:
        test(args)