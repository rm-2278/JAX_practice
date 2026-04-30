import jax
import jax.numpy as jnp
import optax
from flax import nnx

class Encoder(nnx.Module):
    def __init__(self, din, dmid, dlatent, rngs):
        self.fc = nnx.Linear(din, dmid, rngs=rngs)
        self.fc_mu = nnx.Linear(dmid, dlatent, rngs=rngs)
        self.fc_logvar = nnx.Linear(dmid, dlatent, rngs=rngs)
        
    def __call__(self, x):
        x = nnx.relu(self.fc(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class Decoder(nnx.Module):
    def __init__(self, dmid, dout, dlatent, rngs):
        self.fc1 = nnx.Linear(dlatent, dmid, rngs=rngs)
        self.fc2 = nnx.Linear(dmid, dout, rngs=rngs)
    
    def __call__(self, x):
        return nnx.sigmoid(self.fc2(nnx.relu(self.fc1(x))))
    
class VAE(nnx.Module):
    def __init__(self, din, dmid, dlatent, rngs):
        self.encoder = Encoder(din, dmid, dlatent, rngs)
        self.decoder = Decoder(dmid, din, dlatent, rngs)
        
    def __call__(self, x, key):
        mu, logvar = self.encoder(x)
        std = jnp.exp(0.5*logvar)
        z = mu + std * jax.random.normal(key, logvar.shape)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
        
@nnx.jit
def train_step(x, model, optimizer, key):
    def loss_fn(model):
        recon_x, mu, logvar = model(x, key)
        recon_loss = jnp.sum((recon_x - x)**2, axis=-1).mean()
        kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1).mean()
        return recon_loss + kl_loss
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

model = VAE(784, 128, 32, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
key = jax.random.key(0)
x = jax.random.uniform(key, (128, 784))

for i in range(10):
    key, subkey = jax.random.split(key)
    loss = train_step(x, model, optimizer, subkey)
    print(f"Step: {i} | Loss: {loss:.4f}")