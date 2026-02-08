import jax
import jax.numpy as jnp
import flax.linen as nn

class Encoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1)) # Flatten all but batch
        x = nn.Dense(500)(x) # Hidden layer
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim * 2)(x)
        mu, logvar = jnp.split(x, 2, axis=-1)
        return mu, logvar
    
class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(500)(z)
        x = nn.relu(x)
        x = nn.Dense(28*28)(x)
        return x

class VAE(nn.Module):
    latent_dim: int
    
    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder()
        
    def __call__(self, key, x):
        mu, logvar = self.encoder(x)
        eps = jax.random.normal(key, shape=logvar.shape)
        std = jnp.exp(0.5 * logvar) # mult by 0.5 instead of sqrt
        z = mu + std * eps  # Reparametarization trick. Only 1 MC
        x = self.decoder(z)
        x = x.reshape(-1, 28, 28)
        return x, mu, logvar # Return logvar for stability in KL calc
    