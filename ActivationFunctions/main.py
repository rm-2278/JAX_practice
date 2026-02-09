import numpy as np

import jax
import jax.numpy as jnp

from flax import linen as nn


DATASET_PATH = '../data'
CHECKPOINT_PATH = '../checkpoints'

print("Using device:", jax.devices()[0])

class Sigmoid(nn.Module):
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))
    
class Tanh(nn.Module):
    def __call__(self, x):
        return (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))
    
class Relu(nn.Module):
    def __call__(self, x):
        return jnp.maximum(x, 0)

class LeakyRelu(nn.Module):
    alpha: float = 0.1
    def __call__(self, x):
        return jnp.where(x > 0, x, x*self.alpha)

class Elu(nn.Module):
    def __call__(self, x):
        return jnp.where(x > 0, x, jnp.exp(x) - 1)

class Swish(nn.Module):
    def __call__(self, x):
        return x * nn.sigmoid(x)