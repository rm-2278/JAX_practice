import jax
from jax import random, jit, vmap
from jax.scipy.special import logsumexp
from flax import nnx
import jax.numpy as jnp

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

@jit
def initialize_mlp(sizes, key):
    keys = random.split(key, len(sizes))
    def initialize_layer(m, n, key, scale=1e-2):
        key1, key2 = random.split(key)
        return scale * random.normal(key1, (n, m)), scale * random.normal(key2, (n, )) # weights and biases
    return [initialize_layer(n, m, key) for n, m, key in zip(sizes[:-1], sizes[1:], keys)]


key = random.key(0)
sizes = [784, 512, 512, 10]
model = initialize_mlp(sizes, key)

def relu(x):
    return jnp.maximum(x, 0)

def logits(params, input):
    return jnp.dot(params[0], input) + params[1]

def relu_layer(params, input):
    return relu(logits(params, input))

@jit
def forward_pass(params, input_array):
    
    # Loop over the ReLU hidden layers
    for w, b in params[:-1]:
        input_array = relu_layer([w, b], input_array)
    
    # Perform final trafo to logits
    logits = logits(params[-1], input_array)
    
    return logits - logsumexp(logits) # This gives log probability
    
# Make a batched version of the `predict` function
batched_forward_pass = jit(vmap(forward_pass, in_axes=(None, 0), out_axes=0))




def one_hot(input, k):
    return jnp.array(input[:, None] == jnp.arange(k), dtype=jnp.float32) # (D, ) -> (D, k)
    
def loss(params, in_array, target):
    return -jnp.sum(batched_forward_pass(params, in_array) * target) # y_k log p(x_k)

def accuracy(params, data_loader, num_classes):
    acc = 0
    for data, target in data_loader:
        # flatten
        images = jnp.array(data).reshape(data.size(0), 28 * 28)
        targets = one_hot(target, num_classes)
        
        predicted_class = jnp.argmax(batched_forward_pass(params, images), axis=1)
        target_class = jnp.argmax(targets, axis=1)
        
        acc += jnp.sum(predicted_class == target_class)
    return acc / len(data_loader.dataset)
    

@jit
def update(params, x, y, opt_state):
    """Computes the gradient for a batch and update parameters"""
    None
    

# Hyperparameters





def train():
    None