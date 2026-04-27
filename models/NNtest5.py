import jax
import optax
from flax import nnx

class Model(nnx.Module):
    def __init__(self, din, dmid, dout, rngs):
        self.ln = nnx.Linear(din, dmid, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)
        self.fn_out = nnx.Linear(dmid, dout, rngs=rngs)
    
    def __call__(self, x):
        return self.fn_out(self.dropout(nnx.relu(self.bn(self.ln(x)))))
    
@nnx.jit
def step(model, optimizer, x, y):
    loss_fn = lambda model: ((model(x) - y)**2).mean()
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

model = Model(10, 20, 5, nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
key = jax.random.key(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (100, 10))
y = jax.random.normal(k2, (100, 5))

for i in range(10):
    loss = step(model, optimizer, x, y)
    print(loss)