import numpy as np
import torch
import torchvision
import jax
import matplotlib.pyplot as plt

from model import VAE
from train import TrainerModule
from data import load_data


def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    return torch.from_numpy(imgs.copy())

seed = 42
log_dir = "../checkpoints/vae"
batch_size = 100
num_workers = 4

datasets, dataloaders = load_data(seed=seed, batch_size=batch_size, num_workers=num_workers)
train_dataset, val_dataset, test_dataset = datasets
train_dataloader, val_dataloader, test_dataloader = dataloaders

def train_call(key, latent_dim):
    trainer = TrainerModule(latent_dim, train_dataloader, val_dataloader, log_dir=log_dir, seed=seed)
    train_key, eval_key = jax.random.split(key)
    if not trainer.checkpoint_exists():
        trainer.train_model(train_key)
    else:
        trainer.load_model()
    test_loss = trainer.eval_model(eval_key, test_dataloader)
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, test_loss

key = jax.random.key(1234)
model_dict = {}
for latent_dim in [2, 3, 5, 10, 20, 200]:
    key, call_key = jax.random.split(key)
    trainer, test_loss = train_call(call_key, latent_dim)
    model_dict[latent_dim] = {"trainer": trainer, "result": test_loss}
    
def visualize_reconstructions(key, trainer, input_imgs):
    reconst_imgs, _, _ = trainer.model_bd(key, input_imgs)
    imgs = np.stack([input_imgs, reconst_imgs], axis=0).reshape(-1, *input_imgs.shape[1:])
    imgs = imgs[:, None, :, :] # Add channel dim (N, 1, H, W)
    imgs = jax_to_torch(imgs)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(0, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure()
    plt.title(f"Reconstruction using {trainer.latent_dim} dimensions")
    plt.imshow(grid, cmap='gray')
    plt.grid('off')
    plt.show()
    
    
input_imgs = np.stack([train_dataset[i][0] for i in range(4)], axis=0)
for latent_dim in model_dict:
    key, call_key = jax.random.split(key)
    visualize_reconstructions(call_key, model_dict[latent_dim]["trainer"], input_imgs)