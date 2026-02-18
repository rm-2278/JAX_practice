import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from typing import List, Tuple, Sequence, Callable, NamedTuple, Optional, Any
Pytree = Any
from dataclasses import dataclass

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from model import BaseNN

DATASET_PATH = "../data"
CHECKPOINT_PATH = "../checkpoints/optimizers"
print(f"Device: {jax.devices()[0]} used.")


# Dataset pre-processing
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = (img / 255. - 0.2861) / 0.3530 # normalization
    return img

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(sample) for sample in transposed]
    else:
        return np.array(batch)
    
    
@dataclass #generates boilerplate
class TrainConfig:
    dataset_path: str = DATASET_PATH
    batch_size: int = 1024
    num_workers: int = 4


cfg = TrainConfig()


train_dataset = FashionMNIST(root=cfg.dataset_path, train=True, transform=image_to_numpy, download=True)
train_set, val_set = data.random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))
test_set = FashionMNIST(root=cfg.dataset_path, train=False, transform=image_to_numpy, download=True)


batch_size = cfg.batch_size

train_dataloader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, collate_fn=numpy_collate, pin_memory=True, persistent_workers=True)
val_dataloader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, collate_fn=numpy_collate)
test_dataloader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, collate_fn=numpy_collate)

# # Check mean and std
# print(f"Mean: {(train_dataset.data.float() / 255.).mean().item()}")
# print(f"Std: {(train_dataset.data.float() / 255.).std().item()}")

# imgs, _ = next(iter(train_dataloader))
# print(f"Mean: {imgs.mean().item():5.3f}")
# print(f"Std: {imgs.std().item():5.3f}")
# print(f"Max: {imgs.max().item():5.3f}")
# print(f"Min: {imgs.min().item():5.3f}")


act_fn_by_name = {
    "tanh": nn.tanh,
    "relu": nn.relu,
    "identity": lambda x: x
}

def plot_dists(val_dict, color, xlabel, stat="count", use_kde=True):
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns*3, 2.5))
    fig_index = 0
    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(val_dict[key], ax=key_ax, color=color, bins=50, stat=stat, kde=use_kde and (val_dict[key].max() - val_dict[key].min()) > 1e-8)
        key_ax.set_title(f"{key}" + (r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape)>1 else ""))
        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
        fig_index += 1
    fig.subplots_adjust(wspace=0.4)
    return fig

def vizualize_weights(params, color="C0"):
    params, _ = jax.tree_util.tree_flatten(params)
    params = [p.reshape(-1) for p in params if len(p.shape) > 1] # remove biases
    params = jax.device_get(params)
    weights = {f"Layer {layer_idx*2}": p  for layer_idx, p in enumerate(params)}
    
    fig = plot_dists(weights, color=color, xlabel="Weight values")
    fig.suptitle("Weight distribution", y=1.05)
    plt.savefig('image/weight_distribution.png')
    plt.show()
    plt.close()
    
small_dataloader = data.DataLoader(train_set, batch_size=256, shuffle=False, drop_last=True, collate_fn=numpy_collate)
example_images, example_labels = next(iter(small_dataloader))

def visualize_gradient(net, params, color="C0", print_variance=False):
    def loss(params):
        logits = net.apply(params, example_images)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, example_labels).mean()
        return loss
    
    grads = jax.grad(loss)(params)
    grads = jax.device_get(grads)
    grads = jax.tree_util.tree_leaves(grads)
    grads = [g.reshape(-1) for g in grads if g.ndim > 1]
    grads = {f"Layer {layer_idx*2}": g  for layer_idx, g in enumerate(grads)}

    fig = plot_dists(grads, color=color, xlabel="Gradient magnitude")
    fig.suptitle("Gradient distribution", y=1.05)
    plt.savefig("images/gradient_distribution.png")
    plt.show()
    plt.close()
    
    if print_variance:
        for key in sorted(grads.keys()):
            print(f"{key} Variance: {np.var(grads[key])}")

def visualize_activations(net, params, color="C0", print_variance=False):
    _, activations = net.apply(params, example_images, return_activation=True)
    activations = {f"Layer {layer_idx*2}": act.reshape(-1) for layer_idx, act in enumerate(activations[::2])}
    
    fig = plot_dists(activations, color=color, stat="density", xlabel="Activation vals")
    fig.suptitle("Activation distribution", y=1.05)
    plt.savefig("images/activation_distribution.png")
    plt.show()
    plt.close()
    
    
    if print_variance:
        for key in sorted(activations.keys()):
            print(f"{key} Variance: {np.var(activations[key])}")


def init_simple_model(kernel_init, act_fn=act_fn_by_name["identity"]):
    model = BaseNN(act_fn=act_fn, kernel_init=kernel_init)
    params = model.init(random.key(42), example_images)
    return model, params

def get_const_init_func(c=0.0):
    return lambda key, shape, dtype: c*jnp.ones(shape, dtype=dtype)

# # Will give the same gradient for all nodes in intermediate layers
# model, params = init_simple_model(get_const_init_func(0.005))
# visualize_gradient(model, params)
# visualize_activations(model, params, print_variance=True)

# 0.01 will give vanishing activation, 0.1 will give exploding activation
# Gradient explosion when std=2.5
def get_const_var(std=0.1):
    return lambda key, shape, dtype: std*random.normal(key, shape, dtype=dtype)

model, params = init_simple_model(get_const_var(std=0.1))
visualize_gradient(model, params)
visualize_activations(model, params, print_variance=True)
