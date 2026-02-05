import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp

import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, Decoder, AutoEncoder
from train import TrainerModule
from utils import jax_to_torch
from config import TrainConfig

cfg = TrainConfig()

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

# Training for different latent dimensions
def train_call(latent_dim):
    trainer = TrainerModule(
        c_hid=cfg.c_hid,
        latent_dim=latent_dim,
        checkpoint_path=cfg.checkpoint_path,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=cfg.lr,
        seed=cfg.seed,
        prefetch_size=cfg.prefetch_size,
    )
    if not trainer.checkpoint_exists():
        trainer.train_model()
    else:
        trainer.load_model(pretrained=False) # True if downloaded
    test_loss = trainer.eval_model(test_dataloader)
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, test_loss




train_dataset = CIFAR10(root=cfg.dataset_path, train=True, transform=image_to_numpy, download=True)
train_dataset, val_dataset = data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
test_dataset = CIFAR10(root=cfg.dataset_path, train=False, transform=image_to_numpy, download=True)


batch_size = cfg.batch_size

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, collate_fn=numpy_collate, pin_memory=True, persistent_workers=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, collate_fn=numpy_collate)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, collate_fn=numpy_collate)



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

for latent_dim in cfg.latent_dims:
    trainer, test_loss = train_call(latent_dim)
    model_dict[latent_dim] = {"trainer": trainer, "results": test_loss}
    
# latent_dim = [k for k in model_dict]
# results = [model_dict[k]["results"] for k in latent_dim]

# fig = plt.figure(figsize=(6, 4))
# plt.plot(latent_dim, results, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
# plt.xscale("log")
# plt.xticks(latent_dim, labels=latent_dim)
# plt.title("Reconstruction loss vs latent dimension")
# plt.xlabel("Latent dimension")
# plt.ylabel("Reconstruction loss")
# plt.minorticks_off()
# plt.ylim(0, 100)
# os.makedirs("image", exist_ok=True)
# plt.savefig('image/autoencoder.png')


def visualize_reconstructions(trainer, input_imgs):
    # Reconstruct images
    reconst_imgs = trainer.model_bd(input_imgs)
    imgs = np.stack([input_imgs, reconst_imgs], axis=1).reshape(-1, *input_imgs.shape[1:])
    
    imgs = jax_to_torch(imgs)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0) # So matplotlib can recognise
    plt.figure(figsize=(7, 4.5))
    plt.title(f"Reconstructed using {trainer.latent_dim} dimensions")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
    
# input_imgs = np.stack([train_dataset[i][0] for i in range(4)], axis=0) 
# for latent_dim in model_dict:
#     visualize_reconstructions(model_dict[latent_dim]["trainer"], input_imgs)
    
# # Random images
# rng = jax.random.key(123)
# rng, noise_rng = jax.random.split(rng)
# imgs = jax.random.uniform(noise_rng, (2, 32, 32, 3)) * 2 - 1
# visualize_reconstructions(model_dict[64]["trainer"], imgs)


# # Patterned images
# imgs = np.zeros((4, 32, 32, 3)) # jax arrays are immutable
# imgs[1, :, :, 0] = 1
# imgs[2, :, :, :] = -1
# imgs[2, :, :, :] = 1
# xx, yy = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing='ij')
# imgs[3, :, :, 0] = xx #Red
# imgs[3, :, :, 1] = yy #Green
# visualize_reconstructions(model_dict[256]["trainer"], imgs)


# # Reconstruction
# key = jax.random.key(0)
# key, latent_key = jax.random.split(key)
# latent_imgs = jax.random.normal(key, (8, model_dict[256]["trainer"].latent_dim))

# imgs = model_dict[256]["trainer"].model_bd.decoder(latent_imgs)
# imgs = jax_to_torch(imgs)
# grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1), pad_value=0.5)
# grid = grid.permute(1, 2, 0)
# plt.figure(figsize=(7, 4))
# plt.imshow(grid)
# plt.axis('off')
# plt.show()

# # Clustering
trainer = model_dict[64]["trainer"]

def embed_imgs(trainer, dataloader):
    img_list, z_list = [], []
    
    @jax.jit
    def encode(imgs):
        return trainer.model_bd.encoder(imgs)
    
    for imgs, _ in tqdm(dataloader, desc="Encoding images", leave=False): # Remove progress bar afterwards
        z = encode(imgs)
        z_list.append(z)
        img_list.append(imgs)
    
    return (jax.device_get(jnp.concatenate(img_list, axis=0)), jax.device_get(jnp.concatenate(z_list, axis=0)))

train_embeds = embed_imgs(trainer, train_dataloader)
test_embeds = embed_imgs(trainer, test_dataloader)
        
def find_similar_images(img, img_z, key_embeds, K = 8):
    dist = np.linalg.norm(img_z[None] - key_embeds[1], axis=1) # None inserts leading axis
    index = np.argsort(dist)
    
    img_to_show = key_embeds[0][index[:K]]
    img_to_show = np.concatenate([img[None], img_to_show], axis=0)
    img_to_show = torch.from_numpy(img_to_show).permute(0, 3, 1, 2) # np to pytorch
    #plotting
    grid = torchvision.utils.make_grid(img_to_show, nrow=K+1, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0) # torch to plt
    plt.figure(figsize=(12, 3))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
    
    
for i in range(4):
    find_similar_images(train_embeds[0][i], train_embeds[1][i], key_embeds=test_embeds)

# # Clustering with tensorboard

# trainer = model_dict[64]["trainer"]
# writer = SummaryWriter('tensorboard/')

# num_imgs = len(test_dataset)
# writer.add_embedding(test_embeds[1][:num_imgs], #embedding space
#                      metadata=[test_dataset[i][1] for i in range(num_imgs)], #label
#                      label_img=torch.from_numpy(test_embeds[0][:num_imgs] + 1).permute(0, 3, 1, 2)/2.0) #original img

# writer.close()