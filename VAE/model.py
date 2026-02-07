import jax
import jax.numpy as jnp
import flax.linen as nn

class Encoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1)) # Flatten all but batch
        x = nn.Dense(self.latent_dim * 2)(x)
        mu, sigma = jnp.split(x, 2, axis=-1)
        return mu, sigma
    
class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(28*28)(z)
        return x

class VAE(nn.Module):
    latent_dim: int
    
    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder()
        
    def __call__(self, key, x):
        mu, sigma = self.encoder(x)
        eps = jax.random.normal(key, shape=sigma.shape)
        z = mu + sigma * eps  # Reparametarization trick. Only 1 MC
        x = self.decoder(z)
        x = x.reshape(-1, 28, 28)
        return x, mu, sigma
    