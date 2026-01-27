import jax
from jax import random
from jax.scipy.special import logsumexp
from flax import nnx

import torch
from torchvision import datasets, transforms

import time

# Preparing the dataset

batch_size = 100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3801,))
])

training_data = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
        train=True,
        download=True,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True
)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
    train=False,
    transform=transform),
    batch_size=batch_size,
    shuffle=True
)


# Setting up the model

def initialize_mlp(sizes, key):
    keys = random.split(key, len(sizes))
    def initialize_layer(m, n, key, scale=1e-2):
        key1, key2 = random.split(key)
        return scale * random.normal(key1, (n, m)), scale * random.normal(key2, (n, )) # weights and biases
    return [initialize_layer(n, m, key) for n, m, key in zip(sizes[:-1], sizes[1:], keys)]


key = random.key(0)
sizes = [784, 512, 512, 10]
initialize_mlp(sizes, key)