import os
from argparse import ArgumentParser

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler

from dataset import iclevrDataset
from ddpm import conditionalDDPM



def get_random_timesteps(batch_size, total_timesteps, device):
    return torch.randint(0, total_timesteps, (batch_size,)).long().to(device)

def save_checkpoint(model, optimizer, path, epoch):
    save_dir = os.path.join(path, f'checkpoint_{epoch}.pth')
    torch.save({'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}, save_dir)

def load_checkpoint(model, optimizer, path):
    model.load_state_dict(torch.load(path)['model'])
    optimizer.load_state_dict(torch.load(path)['optimizer'])

def train_one_epoch(epoch, model, optimizer, train_loader, loss_function, noise_scheduler, total_timesteps, device):
    model.train()
    train_loss = []
    progress_bar = tqdm(train_loader, desc=f'Epoch: {epoch}', leave=True)
    for i, (x, label) in enumerate(progress_bar):
        batch_size = x.shape[0]
        x, label = x.to(device), label.to(device)
        noise = torch.randn_like(x)
        
        timesteps = get_random_timesteps(batch_size, total_timesteps, device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        output = model(noisy_x, timesteps, label)
        
        loss = loss_function(output, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        progress_bar.set_postfix({'Loss': np.mean(train_loss)})
        
    return np.mean(train_loss)

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--total_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='iclevr')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint.pth')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_freq', type=int, default=2)
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.log_dir}/iclevr_ddpm")
    
    dataset = iclevrDataset(args.dataset, "train")
    train_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers)

    model = conditionalDDPM().to(args.device)
    mse_loss = nn.MSELoss()
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.total_timesteps, beta_schedule=args.beta_schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.resume:
        load_checkpoint(model, optimizer, args.checkpoint)
        
    for epoch in range(args.num_epochs):
        loss = train_one_epoch(epoch, model, optimizer, train_loader, mse_loss, noise_scheduler, args.total_timesteps, args.device)
        writer.add_scalar('Loss/train', loss, epoch)
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, args.save_dir, epoch)
    save_checkpoint(model, optimizer, args.save_dir, args.num_epochs)

if __name__ == '__main__':
    main()