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
