import jax.numpy as jnp
import numpy as np
from jax import jit

jax_array = jnp.array([10, 20])
a1 = jax_array
a1 += 10
print(jax_array, a1) # Out of place

np_array = np.array([10, 20])
a2 = np_array
a2 += 10
print(np_array, a2) # In place


arr = jnp.zeros((4, 4))
arr = arr.at[2, :].set(1)
arr = arr.at[::2, 2:].add(20)

print(arr)
