import jax.numpy as jnp
from jax import random, vmap, jit

key = random.key(0)
key1, key2 = random.split(key)
mat = random.normal(key1, (3, 4))
batched_x = random.normal(key2, (10, 4))

def apply_matrix(x):
    return jnp.dot(mat, x)

def vmap_apply_matrix(batched_x):
    return jnp.stack([apply_matrix(x) for x in batched_x])

@jit
def vmap_apply_matrix_2(batched_x):
    return jnp.dot(mat, batched_x.T)

@jit
def vmap_apply_matrix_3(batched_x):
    return vmap(apply_matrix)(batched_x)


print(vmap_apply_matrix_3(batched_x))

import time

start = time.time()
vmap_apply_matrix_3(batched_x).block_until_ready()
end = time.time()

print(f"{end - start:.6f}")


def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.stack(output)

x = jnp.arange(5)
xs = jnp.stack([x, x])
w = jnp.arange(3)

print(vmap(convolve, in_axes=[0, None])(xs, w))