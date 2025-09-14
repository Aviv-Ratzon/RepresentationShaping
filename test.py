#!/usr/bin/env python3
"""
mnist_action_gan.py

Train a conditional image-to-image model on MNIST:
Input: (mnist_image, one_hot_action) -> Output: image of (digit label + action)
Action range: integers in [-A, A]. One-hot action size = 2*A + 1.
If cyclic=True, label addition wraps modulo 10.
If cyclic=False, datapoints that produce a target outside 0..9 are discarded.

Architecture: U-Net generator + PatchGAN discriminator (pix2pix-like).
Loss: BCE adversarial + L1 reconstruction loss.
"""
import argparse
import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image

# ---------------------------
# Dataset
# ---------------------------
class ActionMNIST(Dataset):
    """
    Returns pairs (input_image, action_onehot, target_image, input_label, target_label).
    - mnist images are normalized to [-1,1].
    - action is sampled uniformly from [-A, A] (inclusive).
    - If cyclic=False, invalid target labels are discarded (dataset length reduced).
    """
    def __init__(self, root: str, train: bool, A: int, cyclic: bool, transform=None, download=True):
        super().__init__()
        assert A >= 0, "A must be non-negative"
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        self.A = A
        self.range_actions = list(range(-A, A + 1))
        self.action_size = len(self.range_actions)  # 2*A + 1
        self.cyclic = cyclic

        # build index of valid pairs if cyclic is False
        self.pairs = []  # list of tuples (mnist_idx, action)
        for idx in range(len(self.mnist)):
            label = int(self.mnist[idx][1])
            for a in self.range_actions:
                tgt = label + a
                if cyclic:
                    # always valid (wrap)
                    self.pairs.append((idx, a))
                else:
                    if 0 <= tgt <= 9:
                        self.pairs.append((idx, a))
        # if user wants fewer samples per epoch, they can use DataLoader's subset/sampler
        print(f"ActionMNIST: {len(self.mnist)} base images, A={A}, cyclic={cyclic} -> {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        mnist_idx, action = self.pairs[i]
        img, label = self.mnist[mnist_idx]
        label = int(label)
        action_idx = self.range_actions.index(action)  # index into one-hot

        if self.cyclic:
            tgt_label = (label + action) % 10
        else:
            tgt_label = label + action  # guaranteed 0..9

        # Load target image: sample a random image from MNIST with that target label
        # For simplicity we draw a random example of the target digit from the dataset.
        # We could alternatively compute deterministic mapping; random is fine for training.
        target_idx = self._sample_index_of_label(tgt_label)
        tgt_img, _ = self.mnist[target_idx]

        # action one-hot vector
        action_onehot = torch.zeros(self.action_size, dtype=torch.float32)
        action_onehot[action_idx] = 1.0

        return img, action_onehot, tgt_img, label, tgt_label

    def _sample_index_of_label(self, label):
        # naive linear scan - okay for small dataset
        # Could cache indices per label for speed.
        for i in range(len(self.mnist)):
            if int(self.mnist[i][1]) == label:
                return i
        # fallback (should not happen)
        return 0


# ---------------------------
# Models
# ---------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class UNetGenerator(nn.Module):
    """
    U-Net like generator for 28x28 grayscale images.
    Input channels = 1 + action_size
    Output channels = 1
    """
    def __init__(self, in_channels: int, out_channels: int = 1, ngf: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, ngf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, 4, 2, 1), nn.BatchNorm2d(ngf*2), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, 3, 1, 1), nn.BatchNorm2d(ngf*4), nn.LeakyReLU(0.2, inplace=True))
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(ngf*4, ngf*4, 3, 1, 1), nn.ReLU(inplace=True))
        # Decoder
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(ngf*4*2, ngf*2, 4, 2, 1), nn.BatchNorm2d(ngf*2), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(ngf*2*2, ngf, 4, 2, 1), nn.BatchNorm2d(ngf), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(ngf*2, out_channels, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        # x: (B, C, 28, 28)
        e1 = self.enc1(x)  # -> (B, ngf, 14, 14)
        e2 = self.enc2(e1)  # -> (B, ngf*2, 7, 7)
        e3 = self.enc3(e2)  # -> (B, ngf*4, 7, 7)
        b = self.bottleneck(e3)  # (B, ngf*4, 7,7)
        d3_in = torch.cat([b, e3], dim=1)  # (B, ngf*8, 7,7)
        d3 = self.dec3(d3_in)  # -> (B, ngf*2, 14,14)
        d2_in = torch.cat([d3, e2], dim=1)  # -> (B, ngf*4, 14,14)
        d2 = self.dec2(d2_in)  # -> (B, ngf, 28,28)
        d1_in = torch.cat([d2, e1], dim=1)  # -> (B, ngf*2, 28,28)
        out = self.dec1(d1_in)  # -> (B,1,28,28)
        return out

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator: input is concatenation of input image and target image (real or generated).
    """
    def __init__(self, in_channels: int = 2, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),  # 28->14
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),  # 14->7
            nn.Conv2d(ndf*2, ndf*4, 3, 1, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),  # 7->7
            nn.Conv2d(ndf*4, 1, 3, 1, 1)  # final patch response (no activation)
        )

    def forward(self, x):
        return self.model(x)  # returns (B,1,H,W)


# ---------------------------
# Training utilities
# ---------------------------
def action_to_spatial(action_onehot: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Expand action one-hot vector (B, action_size) to spatial channels (B, action_size, H, W)
    by repeating values over spatial dims.
    """
    B, S = action_onehot.shape
    return action_onehot.view(B, S, 1, 1).repeat(1, 1, H, W)


def save_samples(gen, dataloader, device, out_dir, action_size, n=8):
    gen.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        batch = next(iter(dataloader))
        imgs, actions, tgts, _, _ = batch
        imgs = imgs.to(device)
        actions = actions.to(device)
        tgts = tgts.to(device)
        action_maps = action_to_spatial(actions, imgs.size(2), imgs.size(3))
        gen_in = torch.cat([imgs, action_maps], dim=1)
        fake = gen(gen_in)
        # concatenate inputs, fake, targets to a grid for visualization
        grid = torch.cat([imgs, fake, tgts], dim=0)
        # denormalize from [-1,1] to [0,1]
        save_image((grid + 1) / 2.0, os.path.join(out_dir, f"samples_{torch.randint(0,100000,()).item()}.png"), nrow=n)
    gen.train()


# ---------------------------
# Main training loop
# ---------------------------
def train(opts):
    device = torch.device("cuda" if (torch.cuda.is_available() and not opts.no_cuda) else "cpu")
    print("Using device:", device)

    # transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])

    dataset = ActionMNIST(root=opts.data_root, train=True, A=opts.A, cyclic=opts.cyclic, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = ActionMNIST(root=opts.data_root, train=False, A=opts.A, cyclic=opts.cyclic, transform=transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=1)

    action_size = dataset.action_size
    print("Action size:", action_size)

    # models
    G = UNetGenerator(in_channels=1 + action_size, out_channels=1, ngf=opts.ngf).to(device)
    D = PatchDiscriminator(in_channels=2, ndf=opts.ndf).to(device)
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    # losses and optimizers
    adversarial_loss = nn.BCEWithLogitsLoss()
    pixel_loss = nn.L1Loss()

    g_optimizer = torch.optim.Adam(G.parameters(), lr=opts.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=opts.lr, betas=(0.5, 0.999))

    real_label = 1.0
    fake_label = 0.0

    os.makedirs(opts.checkpoint_dir, exist_ok=True)
    os.makedirs(opts.sample_dir, exist_ok=True)

    print(">>> Starting training")
    for epoch in range(1, opts.epochs + 1):
        for i, (imgs, actions, tgts, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            tgts = tgts.to(device)
            actions = actions.to(device)

            B, C, H, W = imgs.shape
            action_maps = action_to_spatial(actions, H, W)  # (B, action_size, H, W)
            gen_in = torch.cat([imgs, action_maps], dim=1)  # (B, 1+action_size, H, W)

            # ---------------------
            # Train Discriminator
            # ---------------------
            D.zero_grad()
            # Real pair: input + target
            real_pair = torch.cat([imgs, tgts], dim=1)  # (B,2,H,W)
            out_real = D(real_pair)
            real_labels = torch.full_like(out_real, real_label, device=device)
            loss_d_real = adversarial_loss(out_real, real_labels)

            # Fake pair: input + G(input)
            fake_tgt = G(gen_in)
            fake_pair = torch.cat([imgs, fake_tgt.detach()], dim=1)
            out_fake = D(fake_pair)
            fake_labels = torch.full_like(out_fake, fake_label, device=device)
            loss_d_fake = adversarial_loss(out_fake, fake_labels)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            d_optimizer.step()

            # ---------------------
            # Train Generator
            # ---------------------
            G.zero_grad()
            fake_pair_for_g = torch.cat([imgs, fake_tgt], dim=1)
            out_fake_for_g = D(fake_pair_for_g)
            # generator wants discriminator to label fakes as real
            loss_g_adv = adversarial_loss(out_fake_for_g, real_labels)
            loss_g_l1 = pixel_loss(fake_tgt, tgts) * opts.l1_lambda
            loss_g = loss_g_adv * opts.gan_lambda + loss_g_l1
            loss_g.backward()
            g_optimizer.step()

            if (i + 1) % opts.log_interval == 0:
                print(f"Epoch [{epoch}/{opts.epochs}] Iter [{i+1}/{len(dataloader)}] "
                      f"Loss_D: {loss_d.item():.4f} Loss_G_adv: {loss_g_adv.item():.4f} Loss_G_L1: {loss_g_l1.item():.4f}")

        # save samples and checkpoint each epoch (or every few epochs in a real run)
        if epoch % opts.sample_every == 0 or epoch == 1:
            save_samples(G, val_loader, device, opts.sample_dir, action_size)
            torch.save({'epoch': epoch, 'G_state': G.state_dict(), 'D_state': D.state_dict(),
                        'g_opt': g_optimizer.state_dict(), 'd_opt': d_optimizer.state_dict()},
                       os.path.join(opts.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
    print("Training finished. Models saved in", opts.checkpoint_dir)


# ---------------------------
# Command line / entrypoint
# ---------------------------
from types import SimpleNamespace

def get_opts(
    data_root="./data",
    A=2,
    cyclic=False,
    batch_size=64,
    epochs=20,
    lr=2e-4,
    ngf=64,
    ndf=64,
    l1_lambda=100.0,
    gan_lambda=1.0,
    checkpoint_dir="./checkpoints",
    sample_dir="./samples",
    sample_every=1,
    log_interval=100,
    no_cuda=False
):
    """
    Returns a SimpleNamespace with all training options.
    Can be called from a notebook cell with keyword arguments.
    """
    return SimpleNamespace(
        data_root=data_root,
        A=A,
        cyclic=cyclic,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        ngf=ngf,
        ndf=ndf,
        l1_lambda=l1_lambda,
        gan_lambda=gan_lambda,
        checkpoint_dir=checkpoint_dir,
        sample_dir=sample_dir,
        sample_every=sample_every,
        log_interval=log_interval,
        no_cuda=no_cuda
    )


opts = get_opts()
train(opts)
