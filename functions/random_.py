import jax
import jax.numpy as jnp
from jax import random


key = random.key(0)
for i in range(3):
    new_key, subkey = random.split(key)
    del key # because it is consumed

    val = random.normal(subkey)
    del subkey # again, consumed

    print(val)
    key = new_key


key = random.key(42)
subkeys = random.split(key, 3)
print(random.normal(key, shape=(3,))) # No sequential equivalence
print(jnp.stack([random.normal(subkey) for subkey in subkeys])) # Keys determine the randomness
print(jax.vmap(random.normal)(subkeys))