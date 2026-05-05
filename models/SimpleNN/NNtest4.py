import jax
import optax
from flax import nnx

class Model(nnx.Module):
    def __init__(self, din, dmid, dout, rngs):
        self.ln = nnx.Linear(din, dmid, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)
        self.fc_out = nnx.Linear(dmid, dout, rngs=rngs)
        
    def __call__(self, x):
        return self.fc_out(self.dropout(nnx.relu(self.bn(self.ln(x)))))
    
    
model = Model(4, 10, 2, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


@nnx.jit
def step(x, y, model, optimizer):
    loss_fn = lambda model: ((model(x) - y)**2).mean()
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

key = jax.random.key(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (100, 4))
y = jax.random.normal(k2, (100, 2))

for i in range(20):
    loss = step(x, y, model, optimizer)
    print(loss)