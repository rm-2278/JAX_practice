import numpy as np

import jax
import jax.numpy as jnp

from flax import linen as nn

import matplotlib.pyplot as plt


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
    
act_fn_dict = {
    "sigmoid": Sigmoid,
    "relu": Relu,
    "leakyrelu": LeakyRelu,
    "elu": Elu,
    "swish": Swish
}

def get_grad(act_fn, x):
    # grad requires act_fn to spit a scalar, so vmap batchifies it. 
    #Circumvents the summation trick that is required otherwise.
    return jax.vmap(jax.grad(act_fn))(x)

def visulize_activation_function(act_fn):
    

act_fns = [act_fn() for act_fn in act_fn_dict]
x = np.linspace(-5, 5, 1000)
rows = np.ceil(len(act_fn_dict) / 2.)
fig, ax = plt.