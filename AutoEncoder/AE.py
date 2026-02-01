import os
import json
import math
import numpy as np
from scipy import spatial

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
# import seaborn as sns

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random

from flax import linen as nn
from flax.training import train_state, checkpoints

import optax

import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10
# from torch.utils.tensorboard import SummaryWriter

DATASET_PATH = "../data"
CHECKPOINT_PATH = "../checkpoints/autoencoder"

print("Device: " + str(jax.devices()[0]))


# Dataset pre-processing
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255. * 2 - 1
    return img

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(sample) for sample in transposed]
    else:
        return np.array(batch)


# model architecture
class Encoder(nn.Module):
    c_hid : int
    latent_dim : int
            
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x) # 32x32 -> 16x16, Defaults to padding=Same
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x) # Defaults to padding=Same
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x) # 16x16 -> 8x8
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x) #8x8 -> 4x4
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1) # (batch_size, 2*c_hid*4*4)
        x = nn.Dense(features=self.latent_dim)(x) # preserves the batch_dimension (only applies to the last dimension)
        return x


class Decoder(nn.Module):
    c_out : int
    c_hid : int
    latent_dim : int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*self.c_hid*4*4)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(features=2*self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x) # 4x4 -> 8x8Asymmetric padding
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x) #ConvTranspose works as well
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x) # 8x8 -> 16x16
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_out, kernel_size=(3, 3), strides=(2, 2))(x) #16x16 -> 32x32
        x = nn.tanh(x)
        return x
        
class AutoEncoder(nn.Module):
    c_hid : int
    latent_dim : int
    
    def setup(self): # For explicitly retrieving the sub-modules
        self.encoder = Encoder(self.c_hid, self.latent_dim)
        self.decoder = Decoder(3, self.c_hid, self.latent_dim)
        
    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat



# Loss
def mse_loss(model, params, batch): 
    imgs, _ = batch
    recon_imgs = model.apply({'params': params}, imgs)
    loss = ((imgs - recon_imgs)**2).mean(axis=0).sum() #mean over batch, sum over pixel
    return loss
    
    

train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=image_to_numpy, download=True)
train_dataset, val_dataset = data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
test_dataset = CIFAR10(root=DATASET_PATH, train=False, transform=image_to_numpy, download=True)


batch_size = 256

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=numpy_collate, pin_memory=True, persistent_workers=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)



# key = random.key(0)
# key, enc_key, dec_key = random.split(key, 3)
# encoder = Encoder(c_hid = 32, latent_dim = 128)
# decoder = Decoder(c_hid=32, latent_dim=128, c_out=3)
# img = next(iter(train_dataloader))[0]

# params = encoder.init(enc_key, img)['params']
# latents = encoder.apply({'params': params}, img)
# print(latents.shape)

# dec_params = decoder.init(dec_key, latents)['params']
# result = decoder.apply({'params': dec_params}, latents)
# print(result.shape)

key = random.key(0)
key, ae_key = random.split(key)
autoencoder = AutoEncoder(c_hid=32, latent_dim=128)
imgs = next(iter(train_dataloader))
img = imgs[0]
params = autoencoder.init(ae_key, img)['params']
out = autoencoder.apply({'params': params}, img)
print(out.shape)

print(mse_loss(autoencoder, params, imgs))