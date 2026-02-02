import jax
import jax.numpy as jnp
import numpy as np


my_arrays = [jnp.ones((5, 5)), jnp.ones((10, 2))]

# shapes = jax.tree.map(lambda x: x.shape, my_arrays)
# print([jnp.array(x) for x in shapes])
# zeros = jax.tree.map(lambda x: jnp.zeros(x), [jnp.array(x) for x in shapes]) # convert to jnp so that it will treat as leaf

zeros = jax.tree.map(lambda x: jnp.zeros_like(x), my_arrays)

print(zeros)



