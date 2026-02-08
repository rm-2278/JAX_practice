import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri
import matplotlib.pyplot as plt

from train import TrainerModule
from data import load_data

seed = 42
log_dir = "../checkpoints/vae"
batch_size = 100
num_workers = 4
key = jax.random.key(0)

datasets, dataloaders = load_data(seed=seed, batch_size=batch_size, num_workers=num_workers)
train_dataset, val_dataset, test_dataset = datasets
train_dataloader, val_dataloader, test_dataloader = dataloaders



trainer = TrainerModule(latent_dim=2, train_dataloader=train_dataloader, val_dataloader=val_dataloader, log_dir=log_dir, seed=seed)
train_key, eval_key = jax.random.split(key)
if not trainer.checkpoint_exists():
    trainer.train_model(train_key)
else:
    trainer.load_model()

trainer.model_bd = trainer.model.bind({'params': trainer.state.params})

n = 20
grid_x = np.linspace(0.05, 0.95, n)
grid_y = np.linspace(0.05, 0.95, n)
fig_size = 28
figure = np.zeros((n * fig_size, n * fig_size))

for i, x in enumerate(grid_x):
    for j, y in enumerate(grid_y):
        z_sample = jnp.array([ndtri(x), ndtri(y)])[None, :]
        img = trainer.model_bd.decoder(z_sample)
        img = img.reshape(fig_size, fig_size)
        img = jax.nn.sigmoid(img) # The logit is passed
        figure[i * fig_size: (i+1) * fig_size, j * fig_size: (j+1) * fig_size] = img

plt.figure(figsize=(12, 12))
plt.imshow(figure, cmap='grey_r')
plt.axis('off')
plt.savefig("vae_manifold.png", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()