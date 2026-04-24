import jax
from flax import nnx
import optax

class Model(nnx.Module):
    def __init__(self, din, dmid, dout, rngs):
        self.ln = nnx.Linear(din, dmid, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)
        self.forw = nnx.Linear(dmid, dout, rngs=rngs)
        
        
    def __call__(self, x):
        return self.forw(self.dropout(self.ln(x)))
        
        
model = Model(4, 10, 2, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, tx=optax.adam(1e-3), wrt=nnx.Param)


@nnx.jit
def step(model, optimizer, x, y):
    loss_fn = lambda model: ((model(x) - y)**2).mean()
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss


key = jax.random.key(0)
key1, key2 = jax.random.split(key)
target = jax.random.normal(key1, (10, 4))
label = jax.random.normal(key2, (10, 2))

for i in range(5):
    key = nnx.Rngs(dropout=0)
    loss = step(model, optimizer, target, label)
    print(loss)