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
from evaluator import evaluation_model

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--best_ckpt_path", type=str, help="the best checkpoint's path")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--data_dir', type=str, default='iclevr', help='image file of dataset')
parser.add_argument('--output_dir', type=str, default='Result_DC/', help='output directory of results')
parser.add_argument('--model_dir', type=str, default='checkpoint_DC/', help='output directory of generator')
parser.add_argument('--test_dir', type=str, default='Test_DC/')

opt = parser.parse_args()
os.makedirs(opt.output_dir, exist_ok=True)
os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(opt.test_dir, exist_ok=True)
print(opt)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

dataset = iclevrDataset(opt.data_dir, "train")
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.n_cpu)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, n_classes):
        super(Generator, self).__init__()
        self.nz = nz
        self.class_embedding = nn.Sequential(
            nn.Linear(n_classes, nz),
            nn.LeakyReLU(0.2, inplace=True)
        )
        def dconv_bn_relu(in_dim, out_dim, kernel_size, stride, padding):
            return [
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True)
            ]
            
        self.model = nn.Sequential(
            *dconv_bn_relu(2 * nz, ngf * 8, 4, 1, 0),
            *dconv_bn_relu(ngf * 8, ngf * 4, 4, 2, 1),
            *dconv_bn_relu(ngf * 4, ngf * 2, 4, 2, 1),
            *dconv_bn_relu(ngf * 2, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x, labels):
        x = x.view(-1, self.nz, 1, 1)
        class_embedding = self.class_embedding(labels).reshape(-1, self.nz, 1, 1)
        x = torch.cat((x, class_embedding), dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_classes):
        super(Discriminator, self).__init__()
        self.class_embedding = nn.Linear(n_classes, 64 * 64)
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
    def forward(self, x, labels):
        labels = labels.float()
        class_embedding = self.class_embedding(labels).view(-1, 1, 64, 64)
        x = torch.cat((x, class_embedding), dim=1)
        return self.model(x)


# Loss functions

def adversarial_loss_d(r_logit, f_logit):
    r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
    f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
    return r_loss, f_loss

def adversarial_loss_g(f_logit):
    f_loss = - f_logit.mean()
    return f_loss

# Initialize generator and discriminator
nc = 3  # number of channels
nz = 100 # size of latent vector
ngf = 64 # size of feature maps in generator
ndf = 64 # size of feature maps in discriminator
generator = Generator(nz, ngf, nc, 24)
discriminator = Discriminator(nc, ndf, 24)

# if cuda:
generator.to(device)
discriminator.to(device)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def sample_image(n_row, labels, epoch, acc):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    z = torch.randn(n_row, opt.latent_dim).to(device)
    
    gen_imgs = generator(z, labels)
    path = os.path.join(opt.output_dir, '{}_{:.4f}.png'.format(epoch, acc))
    gen_imgs.data = (gen_imgs.data+1)/2
    save_image(gen_imgs.data, path, nrow=8)


def save(path):
    torch.save({
        'G': generator.state_dict(),
        'D': discriminator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
    }, path)

def load(path):
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['G'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    discriminator.load_state_dict(checkpoint['D'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])

def test(generator, epoch, eval_model):
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
        path = os.path.join(opt.test_dir, '{}_test_{:.4f}.png'.format(epoch, acc))
        gen_imgs = (gen_imgs+1)/2
        save_image(gen_imgs, path, nrow=8)

        labels = next(iter(new_test_dataloader))
        z = torch.randn(32, opt.latent_dim).to(device)
        labels = labels.to(device)
        gen_imgs = generator(z, labels)
        acc = eval_model.eval(gen_imgs, labels)
        new_test_acc = acc
        path = os.path.join(opt.test_dir, '{}_new_test_{:.4f}.png'.format(epoch, acc))
        gen_imgs = (gen_imgs+1)/2
        save_image(gen_imgs, path, nrow=8)
    return test_acc, new_test_acc

# ----------
#  Training
# ----------

eval_model = evaluation_model(device)
test_best_acc = [0, 0]
for epoch in range(opt.n_epochs):
    gloss = 0
    dloss = 0
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="DCGAN")
    acc = 0
    generator.train()
    discriminator.train()
    for i, (imgs, labels) in trange:
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        # generate fake images
        z = torch.randn(batch_size, opt.latent_dim).to(device)
        gen_imgs = generator(z, labels)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        real_pred = discriminator(imgs, labels)
        fake_pred = discriminator(gen_imgs.detach(), labels) # no back-propagation so use detach
        d_real_loss, d_fake_loss = adversarial_loss_d(real_pred, fake_pred) 
        d_loss = d_real_loss + d_fake_loss
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        

        # -----------------
        #  Train Generator
        # -----------------

        validity = discriminator(gen_imgs, labels)
        g_loss = adversarial_loss_g(validity)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        gloss += (g_loss.item())
        dloss += d_real_loss.item() + d_fake_loss.item()
        acc += eval_model.eval(gen_imgs.detach(), labels.detach())
        
        trange.set_postfix({"epoch":"{}".format(epoch),"g_loss":"{0:.5f}".format(gloss / (i + 1)), "d_loss":"{0:.5f}".format(dloss / (i + 1)), "acc":"{0:.5f}".format(acc / (i + 1))})

    sample_image(batch_size, labels, epoch, acc / len(trange))
    
    test_acc, new_test_acc = test(generator, epoch, eval_model)
    if test_acc > test_best_acc[0] and new_test_acc > test_best_acc[1]:
        test_best_acc = [test_acc, new_test_acc]
        save(opt.model_dir + f"model_test_{test_acc:.4f}_{new_test_acc:.4f}.pth")





