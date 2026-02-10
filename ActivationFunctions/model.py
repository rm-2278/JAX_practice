from typing import Sequence
from flax import linen as nn
from jax import random
import jax.numpy as jnp




init_fun = lambda x: (lambda rng, shape, dtype: random.uniform(rng, shape, dtype, minval=-1/jnp.sqrt(x.shape[1]), maxval=1/jnp.sqrt(x.shape[1])))

class NN(nn.Module):
    act_fn: nn.Module
    num_classes: int = 10
    hidden_sizes: Sequence = (512, 256, 256, 128)
    
    @nn.compact
    def __call__(self, x, return_activations=False):
        x = x.reshape(x.shape[0], -1)
        activations = []
        
        for hid in self.hidden_sizes:
            x = nn.Dense(hid, kernel_init=init_fun(x), bias_init=init_fun(x))(x)
            activations.append(x)
            x = self.act_fn(x)
            activations.append(x)
        x = nn.Dense(self.num_classes, kernel_init=init_fun(x), bias_init=init_fun(x))(x)
        
        
        return x if not return_activations else (x, activations)