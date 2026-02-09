import jax
import jax.numpy as jnp

def hybrid_layer(x, mode):
    # Condition A: Hyperparameter-based (Requires a regular value or concrete tracer)
    val = jax.lax.cond(mode==0, lambda x: jnp.mean(x), lambda x: jnp.max(x), x)
    
    # Condition B: Data-dependent (Requires jax.lax.cond if abstract)
    return jax.lax.cond(x.sum() > 0, lambda val: val * 2.0, lambda val: val * -1.0, val)
    
def outer_func(x):
    # We want the gradient of the hybrid layer at a specific point
    return jax.grad(hybrid_layer, argnums=0)(x, 0)

jitted_grad = jax.jit(outer_func) # Outer jit converts to abstract tracers
result = jitted_grad(jnp.array([1.0, 2.0]))
print(result)