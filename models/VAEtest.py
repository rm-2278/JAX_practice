import jax
import jax.numpy as jnp
import optax
from flax import nnx

class Encoder(nnx.Module):
    def __init__(self, din, dmid, dlatent, rngs):
        self.fc1 = nnx.Linear(din, dmid, rngs=rngs)
        # The encoder outputs TWO things: a mean and a log-variance
        self.fc_mu = nnx.Linear(dmid, dlatent, rngs=rngs)
        self.fc_logvar = nnx.Linear(dmid, dlatent, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nnx.Module):
    def __init__(self, dlatent, dmid, dout, rngs):
        self.fc1 = nnx.Linear(dlatent, dmid, rngs=rngs)
        self.fc2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, z):
        z = nnx.relu(self.fc1(z))
        # Sigmoid ensures the output is between 0 and 1 (like pixel values)
        return nnx.sigmoid(self.fc2(z))

class VAE(nnx.Module):
    def __init__(self, din, dmid, dlatent, rngs):
        self.encoder = Encoder(din, dmid, dlatent, rngs)
        self.decoder = Decoder(dlatent, dmid, din, rngs)

    def reparameterize(self, mu, logvar, key):
        # Convert log variance to standard deviation
        std = jnp.exp(0.5 * logvar)
        # Sample random noise using the explicitly passed key
        eps = jax.random.normal(key, logvar.shape)
        return mu + eps * std

    def __call__(self, x, key):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, key)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

@nnx.jit
def train_step(model, optimizer, x, key):
    def loss_fn(model):
        recon_x, mu, logvar = model(x, key)
        
        # 1. Reconstruction Loss (Mean Squared Error)
        recon_loss = ((recon_x - x)**2).sum(axis=-1).mean()
        
        # 2. KL Divergence Loss
        kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1).mean()
        
        # Total Loss
        return recon_loss + kl_loss

    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

# --- Setup and Training Loop ---

rngs = nnx.Rngs(0)
# Example dimensions: 784 (e.g., flattened 28x28 image), 256 hidden, 32 latent
model = VAE(din=784, dmid=256, dlatent=32, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# Main PRNG key for the loop
key = jax.random.key(1)
# Dummy image batch (values bounded between 0 and 1 to match sigmoid output)
x_dummy = jax.random.uniform(key, (128, 784)) 

for i in range(10):
    # CRITICAL: Split the key every step to ensure new random noise
    key, subkey = jax.random.split(key)
    
    loss = train_step(model, optimizer, x_dummy, subkey)
    print(f"Step {i} | Loss: {loss:.4f}")