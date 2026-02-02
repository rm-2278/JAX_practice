import jax
from flax import linen as nn

# model architecture
class Encoder(nn.Module):
    c_hid : int
    latent_dim : int
            
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x) # 32x32 -> 16x16, Defaults to padding=Same
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x) # Defaults to padding=Same
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x) # 16x16 -> 8x8
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x) #8x8 -> 4x4
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1) # (batch_size, 2*c_hid*4*4)
        x = nn.Dense(features=self.latent_dim)(x) # preserves the batch_dimension (only applies to the last dimension)
        return x


class Decoder(nn.Module):
    c_out : int
    c_hid : int
    latent_dim : int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*self.c_hid*4*4)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(features=2*self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x) # 4x4 -> 8x8Asymmetric padding
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3))(x) #ConvTranspose works as well
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x) # 8x8 -> 16x16
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.c_out, kernel_size=(3, 3), strides=(2, 2))(x) #16x16 -> 32x32
        x = nn.tanh(x)
        return x
        
class AutoEncoder(nn.Module):
    c_hid : int
    latent_dim : int
    
    def setup(self): # For explicitly retrieving the sub-modules
        self.encoder = Encoder(self.c_hid, self.latent_dim)
        self.decoder = Decoder(3, self.c_hid, self.latent_dim)
        
    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
