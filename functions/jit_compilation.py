import jax
import jax.numpy as jnp
from jax import jit

def normalize(x, mode):
    print(f"Computing normalization for {mode}")
    if mode == 'standard':
        return ((x - jnp.mean(x)) / jnp.std(x))
    if mode == "minmax":
        return ((x - jnp.min(x)) / (jnp.max(x) - jnp.min(x)))
    else:
        return x
    
jitted_normalize = jit(normalize, static_argnums=1)

data = jnp.array([1, 2, 3, 4])
# Run 1
jitted_normalize(data, 'standard')
# Run 2
jitted_normalize(data, 'standard')
# Run 3
jitted_normalize(data, 'minmax')