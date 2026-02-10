import math
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
    
class ReLU(nn.Module):
    def __call__(self, x):
        return jnp.maximum(x, 0)

class LeakyReLU(nn.Module):
    alpha: float = 0.1
    def __call__(self, x):
        return jnp.where(x > 0, x, x*self.alpha)

class ELU(nn.Module):
    def __call__(self, x):
        return jnp.where(x > 0, x, jnp.exp(x) - 1)

class Swish(nn.Module):
    def __call__(self, x):
        return x * nn.sigmoid(x)
    
act_fn_dict = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish
}

def get_grad(act_fn, x):
    # grad requires act_fn to spit a scalar, so vmap batchifies it. 
    #Circumvents the summation trick that is required otherwise.
    return jax.vmap(jax.grad(act_fn))(x)

def visualize_activation_function(act_fn, ax, x):
    y = act_fn(x)
    y_grad = get_grad(act_fn, x)
    
    ax.plot(x, y, linewidth=2, label="ActFn")
    ax.plot(x, y_grad, linewidth=2, label="Gradient")
    ax.set_title(act_fn.__class__.__name__)
    ax.legend()
    ax.set_ylim(-1.5, x.max())

act_fns = [act_fn() for act_fn in act_fn_dict.values()]
x = np.linspace(-5, 5, 1000)
rows = math.ceil(len(act_fn_dict) / 3.)
fig, ax = plt.subplots(rows, 3, figsize=(4*3, 4*rows))
for i, act_fn in enumerate(act_fns):
    visualize_activation_function(act_fn, ax[divmod(i, 3)], x)
    
plt.subplots_adjust(hspace=0.3)
plt.savefig("activation_functions.png")
plt.show()