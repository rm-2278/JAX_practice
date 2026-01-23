import jax
import jax.numpy as jnp
from jax import jit

x = jnp.array([1, 2, 3])

@jit
def f(x):
    print(x)
    return x

@jit
def debugging_f(x):
    # jax.debug.print(x) # Not possible 
    jax.debug.print("{x}", x=x)
    return x

print(f(x))
print(debugging_f(x))

