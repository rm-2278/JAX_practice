import jax
import flax.linen as nn

class Encoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.latent_dim)(x)
        mu = x[:self.latent_dim]
        sigma = x[self.latent_dim:]
        return mu, sigma
    
class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        x = nn.Dense(28*28*3)(z)
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
        return x, mu, sigma
    