import argparse
import os
import numpy as np
import math
import random
import pdb

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from dataset import iclevrDataset
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn.utils import spectral_norm
from evaluator import evaluation_model

parser = argparse.ArgumentParser()
parser.add_argument("--best_ckpt_path", type=str, help="the best checkpoint's path")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument('--data_dir', type=str, default='iclevr', help='image file of dataset')
parser.add_argument('--test_dir', type=str, default='Test', help='output directory of results')

opt = parser.parse_args()
os.makedirs(opt.test_dir, exist_ok=True)
print(opt)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self, z_dim=100, c_dim=100, dim=256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        def dconv_bn_relu(in_dim, out_dim, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.ls = nn.Sequential(
            dconv_bn_relu(z_dim + c_dim, dim * 8, 4, 1, 0),  # (N, dim * 8, 4, 4)
            dconv_bn_relu(dim * 8, dim * 4, 4, 2, 1),  # (N, dim * 4, 8, 8)
            dconv_bn_relu(dim * 4, dim * 2, 4, 2, 1),  # (N, dim * 2, 16, 16)
            dconv_bn_relu(dim * 2, dim * 1, 4, 2, 1),  # (N, dim, 32, 32)
            nn.ConvTranspose2d(dim, 3, 4, 2, 1),  # (N, 3, 64, 64)
            nn.Tanh()  # (N, 3, 64, 64)
        )
        
        self.label_emb = nn.Sequential(
            nn.Linear(24, c_dim),
            nn.ReLU()
        )
    def forward(self, z, c):
        z = z.view(z.size(0), z.size(1), 1, 1)
        c = self.label_emb(c).view(z.size(), self.c_dim, 1, 1)
        x = torch.cat([z, c], 1)
        x = self.ls(x)
        return x

generator = Generator().to(device)

def load(path):
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['G'])

def test(generator, eval_model):
    generator.eval()
    test_dataset = iclevrDataset(opt.data_dir, "test")
    new_test_dataset = iclevrDataset(opt.data_dir, "new_test")
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=opt.n_cpu)
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=32, num_workers=opt.n_cpu)
    test_acc, new_test_acc = 0, 0
    with torch.no_grad():
        labels = next(iter(test_dataloader))
        z = torch.randn(32, opt.latent_dim).to(device)
        labels = labels.to(device)
        gen_imgs = generator(z, labels)
        acc = eval_model.eval(gen_imgs, labels)
        test_acc = acc
        path = os.path.join(opt.test_dir, '{}_test_{:.4f}.png'.format(opt.best_ckpt_path.replace('/', '_'), acc))
        gen_imgs = (gen_imgs+1)/2
        save_image(gen_imgs, path, nrow=8)

        labels = next(iter(new_test_dataloader))
        z = torch.randn(32, opt.latent_dim).to(device)
        labels = labels.to(device)
        gen_imgs = generator(z, labels)
        acc = eval_model.eval(gen_imgs, labels)
        new_test_acc = acc
        path = os.path.join(opt.test_dir, '{}_new_test_{:.4f}.png'.format(opt.best_ckpt_path.replace('/', '_'), acc))
        gen_imgs = (gen_imgs+1)/2
        save_image(gen_imgs, path, nrow=8)
    return test_acc, new_test_acc

load(opt.best_ckpt_path)
eval_model = evaluation_model(device)

test(generator, eval_model)