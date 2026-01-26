import numpy as onp
import jax.numpy as np
import jax
from jax import grad, jit, vmap, value_and_grad
from jax import random
from time import time as timer

key = random.key(1)
x = random.uniform(key, (10000, 10000))

def relu(x):
    return np.maximum(x, 0)

def deriv_relu(x):
    return (relu(x + 1e-3) - relu(x - 1e-3)) / relu(2e-3)

relu(x)

s = timer()
print(deriv_relu(2.))
e = timer()

s = timer()
print(jit(grad(jit(relu)))(2.))
e = timer()
print(f"{e-s:.6f}")

print(f"{e-s:.6f}")

