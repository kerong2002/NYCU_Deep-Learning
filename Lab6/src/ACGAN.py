import argparse
import os
import numpy as np
import math
import random
import pdb

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from dataset import iclevrDataset

from torch.nn.utils import spectral_norm
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch

from evaluator import evaluation_model

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
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
parser.add_argument('--output_dir', type=str, default='Result_AC_300/', help='output directory of results')
parser.add_argument('--model_dir', type=str, default='checkpoint_AC_300/', help='output directory of generator')
parser.add_argument('--test_dir', type=str, default='Test_AC_300/')

opt = parser.parse_args()
os.makedirs(opt.output_dir, exist_ok=True)
os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(opt.test_dir, exist_ok=True)
print(opt)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = iclevrDataset(opt.data_dir, "train")
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.n_cpu)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ndf = 16
        self.ncf = 100
        def discriminator_block(in_dim, out_dim, kernel_size, stride, padding, bn=True):
            block = []
            block.append(spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)))
            if bn: block.append(nn.BatchNorm2d(out_dim))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, self.ndf, 3, 2, 1, bn=False),
            *discriminator_block(self.ndf, self.ndf * 2, 3, 1, 0),
            *discriminator_block(self.ndf * 2, self.ndf * 4, 3, 2, 1),
            *discriminator_block(self.ndf * 4, self.ndf * 8, 3, 1, 0),
            *discriminator_block(self.ndf * 8, self.ndf * 16, 3, 2, 1),
            *discriminator_block(self.ndf * 16, self.ndf * 32, 3, 1, 0),
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(5 * 5 * self.ndf * 32, 1),
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(5 * 5 * self.ndf * 32, 24),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions

auxiliary_loss = torch.nn.BCELoss()

def adversarial_loss_d(r_logit, f_logit):
    r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
    f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
    return r_loss, f_loss

def adversarial_loss_g(f_logit):
    f_loss = - f_logit.mean()
    return f_loss

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


# if cuda:
generator.to(device)
discriminator.to(device)

auxiliary_loss.to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

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
        gen_imgs.data = (gen_imgs.data+1)/2
        save_image(gen_imgs.data, path, nrow=8)

        labels = next(iter(new_test_dataloader))
        z = torch.randn(32, opt.latent_dim).to(device)
        labels = labels.to(device)
        gen_imgs = generator(z, labels)
        acc = eval_model.eval(gen_imgs, labels)
        new_test_acc = acc
        path = os.path.join(opt.test_dir, '{}_new_test_{:.4f}.png'.format(epoch, acc))
        gen_imgs.data = (gen_imgs.data+1)/2
        save_image(gen_imgs.data, path, nrow=8)
    return test_acc, new_test_acc
# ----------
#  Training
# ----------
test_best_acc = [0, 0]
eval_model = evaluation_model()
for epoch in range(opt.n_epochs):

    gloss = 0
    dloss = 0
    acc = 0
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="ACGAN_300")
    generator.train()
    discriminator.train()
    for i, (imgs, labels) in trange:
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        

        z = torch.randn(batch_size, opt.latent_dim).to(device)
        gen_imgs = generator(z, labels)
        
        real_pred, real_label = discriminator(imgs)
        real_auxi_loss = auxiliary_loss(real_label, labels)

        fake_pred, fake_label = discriminator(gen_imgs.detach())
        fake_auxi_loss = auxiliary_loss(fake_label, labels)

        d_fake_loss, d_real_loss = adversarial_loss_d(real_pred, fake_pred) 

        d_loss = d_real_loss + d_fake_loss + 300 * (real_auxi_loss + fake_auxi_loss) / 2
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        validity, pred_label = discriminator(gen_imgs)
        g_loss = adversarial_loss_g(validity) + 300 * auxiliary_loss(pred_label, labels)
        
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