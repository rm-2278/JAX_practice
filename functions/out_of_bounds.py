import jax
import jax.numpy as jnp

print(jnp.arange(10.0)[11])
print(jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan))
