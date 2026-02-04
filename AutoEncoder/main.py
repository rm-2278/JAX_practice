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
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, Decoder, AutoEncoder


DATASET_PATH = "../data"
CHECKPOINT_PATH = "../checkpoints/autoencoder"

print("Device: " + str(jax.devices()[0]))


# Dataset pre-processing
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255. * 2 - 1
    return img

def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    return torch.from_numpy(imgs).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(sample) for sample in transposed]
    else:
        return np.array(batch)

# Loss
def mse_loss(model, params, batch): 
    imgs, _ = batch
    recon_imgs = model.apply({'params': params}, imgs)
    loss = ((imgs - recon_imgs)**2).mean(axis=0).sum() #mean over batch, sum over pixel
    return loss

class GenerateCallback:
    def __init__(self, input_imgs, every_n_epochs=1):
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs
        
    def log_generation(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            reconst_img = model.apply({'params': state.params}, self.input_imgs)
            # Move to device
            reconst_img = jax.device_get(reconst_img)
            
            # Save imgs
            imgs = np.stack([self.input_imgs, reconst_img], axis=1).reshape(-1, *reconst_img.shape[1:])
            imgs = jax_to_torch(imgs)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1, 1))
            # add to logger
            logger.add_image("Reconstructions", grid, global_step=epoch)

# Training functionality
class TrainerModule:
    def __init__(self, c_hid, latent_dim, lr=1e-3, seed=42):
        self.c_hid = c_hid
        self.latent_dim = latent_dim
        self.lr = lr
        self.seed = seed
        
        self.model = AutoEncoder(c_hid, latent_dim)
        
        self.log_dir = os.path.abspath( # orbax tensorstore refuse relative path
            os.path.join(CHECKPOINT_PATH, f'cifar10_{latent_dim}')
        )
        
        self.example_imgs = next(iter(val_dataloader))[0][:8] # For initialising generator, model
        self.generator_callback = GenerateCallback(self.example_imgs, every_n_epochs=50)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        
        self.create_functions()
        
        self.init_model()
        
    def create_functions(self): # Create jitted functions
        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(self.model, params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params) # Or argnums
            state = state.apply_gradients(grads=grads)
            return state, loss
        self.train_step = jax.jit(train_step)
        
        def eval_step(state, batch):
            return mse_loss(self.model, state.params, batch)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        rngs = jax.random.key(self.seed)
        rngs, init_rngs = jax.random.split(rngs)
        params = self.model.init(init_rngs, self.example_imgs)['params']
        lr_schedular = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=100,
            decay_steps=500*len(train_dataloader),
            end_value=1e-5
        )
        optimizer = optax.chain(optax.clip(1), optax.adam(lr_schedular))
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
    
    def train_model(self, num_epochs=500):
        best_loss = 1e6
        for epoch_idx in tqdm(range(num_epochs)):
            self.train_epoch(epoch = epoch_idx)
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(val_dataloader)
                self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_model(step=epoch_idx)
                self.generator_callback.log_generation(self.model, self.state, self.logger, epoch_idx)
                self.logger.flush()
        
    def train_epoch(self, epoch):
        losses = []
        for batch in train_dataloader:
            self.state, loss = self.train_step(self.state, batch)
            losses.append(loss)
        losses = np.stack(jax.device_get(losses))
        avg_loss = losses.mean()
        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
        

    def eval_model(self, dataloader):
        losses = []
        batch_sizes = []
        for batch in dataloader:
            loss = self.eval_step(self.state, batch)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss
    
    def save_model(self, step=0): # Save model during training
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, step=step, target=self.state.params, prefix=f'cifar10_{self.latent_dim}_')
    
    def load_model(self, pretrained=False): # When loading trained or pretrained model
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f'cifar10_{self.latent_dim}_')
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)
    
    def checkpoint_exists(self):
        return checkpoints.latest_checkpoint(self.log_dir, prefix=f'cifar10_{self.latent_dim}_') is not None
        # return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'cifar10_{self.latent_dim}.ckpt'))



# Training for different latent dimensions
def train_call(latent_dim):
    trainer = TrainerModule(c_hid=32, latent_dim=latent_dim)
    if not trainer.checkpoint_exists():
        trainer.train_model()
    else:
        trainer.load_model(pretrained=False) # True if downloaded
    test_loss = trainer.eval_model(test_dataloader)
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, test_loss




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



# key = random.key(0)
# key, ae_key = random.split(key)
# autoencoder = AutoEncoder(c_hid=32, latent_dim=128)
# imgs = next(iter(train_dataloader))
# img = imgs[0]
# params = autoencoder.init(ae_key, img)['params']
# out = autoencoder.apply({'params': params}, img)
# print(out.shape)

# print(mse_loss(autoencoder, params, imgs))



# Storing results
model_dict = {}

for latent_dim in [64, 128, 256, 384, 512]:
    trainer, test_loss = train_call(latent_dim)
    model_dict[latent_dim] = {"trainer": trainer, "results": test_loss}
    
latent_dim = [k for k in model_dict]
results = [model_dict[k]["results"] for k in latent_dim]

fig = plt.figure(figsize=(6, 4))
plt.plot(latent_dim, results, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
plt.xscale("log")
plt.xticks(latent_dim, labels=latent_dim)
plt.title("Reconstruction loss vs latent dimension")
plt.xlabel("Latent dimension")
plt.ylabel("Reconstruction loss")
plt.minorticks_off()
plt.ylim(0, 100)
os.makedirs("image", exist_ok=True)
plt.savefig('image/autoencoder.png')