import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        self.writer = SummaryWriter("logs/")
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)
    def train_one_epoch(self, train_loader, epoch, args):
        losses = []
        progress = tqdm(enumerate(train_loader))
        self.model.train()
        for i, x in progress:
            x = x.to(args.device)
            logits, z_indices = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            loss.backward()
            losses.append(loss.item())
            if i % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            progress.set_description_str(f"epoch: {epoch} / {args.epochs}, iter: {i} / {len(train_loader)}, loss: {np.mean(losses)}")
        self.writer.add_scalar("loss/train", np.mean(losses), epoch)
        return np.mean(losses)
    
    def eval_one_epoch(self, val_loader, epoch, args):
        losses = []
        progress = tqdm(enumerate(val_loader))
        self.model.eval()
        with torch.no_grad():
            for i, x in progress:
                x = x.to(args.device)
                logits, z_indices = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
                losses.append(loss.item())
                progress.set_description_str(f"val_loss: {np.mean(losses)}")
        self.writer.add_scalar("loss/val", np.mean(losses), epoch)
        return np.mean(losses)

    def configure_optimizers(self, args):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scheduler = None
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    best_train = np.inf
    best_val = np.inf
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args)
        
        if train_loss < best_train:
            best_train = train_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/best_train.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/best_val.pth")
        