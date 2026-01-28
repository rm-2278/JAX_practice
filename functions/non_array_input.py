import jax
import jax.numpy as jnp

# print(jnp.sum([1, 2, 3]))

def permissive_sum(x):
    return jnp.sum(jnp.array(x))

x = list(range(10))
print(jax.make_jaxpr(permissive_sum)(x)) # Shows 10 variables independently processed