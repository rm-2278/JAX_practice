import jax
import jax.numpy as jnp
from jax import jit, grad

def simple_graph(x):
    x = x + 2
    x = x**2
    x = x + 3
    y = x.mean()
    return y

inp = jnp.arange(3, dtype=jnp.float32)
print(inp)
print(jax.make_jaxpr(grad(simple_graph))(inp))