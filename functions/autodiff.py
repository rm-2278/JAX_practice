import jax
import jax.numpy as jnp
from jax import jit, vmap, grad


def forward(W, b, inputs):
    return jax.nn.sigmoid(jnp.dot(inputs, W) + b)

def loss(W, b, inputs, targets):
    pred = forward(W, b, inputs)
    prob = pred*targets + (1-pred)*(1-targets)
    return -jnp.sum(jnp.log(prob)) # Take log here


key = jax.random.key(0)

inputs = jnp.array([[0.52, 1.12,  0.77],
                    [0.88, -1.08, 0.15],
                    [0.52, 0.06, -1.30],
                    [0.74, -2.49, 1.39]])
targets = jnp.array([True, False, False, True])

key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.uniform(W_key, (3,))
b = jax.random.uniform(b_key, ())

loss_val, grads = jax.value_and_grad(loss, argnums=(0, 1))(W, b, inputs, targets)
print(loss_val)
print(grads)
eps = 1e-4
b_loss_numerical = (loss(W, b+eps/2, inputs, targets) - loss(W, b-eps/2, inputs, targets)) / eps
print(b_loss_numerical)

