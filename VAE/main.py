import numpy as np
import torch
import torchvision
import jax
import matplotlib.pyplot as plt

from model import VAE
from train import TrainerModule


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32) / 255.0
    return (img >= 0.5).astype(np.float32) # Binarize

def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    return torch.from_numpy(imgs.copy())

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch, axis=0)
    elif isinstance(batch[0], (tuple, list)):
        return [numpy_collate(x) for x in zip(*batch)]
    else:
        return np.array(batch)

seed = 42

train_dataset = torchvision.datasets.MNIST("../data", train=True, transform=image_to_numpy, download=True)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths=[50000, 10000], generator=torch.Generator().manual_seed(seed))
test_dataset = torchvision.datasets.MNIST("../data", train=False, transform=image_to_numpy, download=True)

batch_size = 100
num_workers = 4

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=numpy_collate, pin_memory=True, persistent_workers=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=numpy_collate)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=numpy_collate)


def train_call(key, latent_dim):
    trainer = TrainerModule(latent_dim, train_dataloader, val_dataloader, seed=seed)
    train_key, eval_key = jax.random.split(key)
    trainer.train_model(train_key)
    test_loss = trainer.eval_model(eval_key, test_dataloader)
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, test_loss

key = jax.random.key(1234)
model_dict = {}
for latent_dim in [3, 5, 10, 20, 200]:
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