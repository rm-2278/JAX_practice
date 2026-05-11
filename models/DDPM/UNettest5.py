import jax
import jax.numpy as jnp
import optax
from flax import nnx

class SinusoidalEmbedding(nnx.Module):
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, time):
        half_dim = self.dim // 2
        embedding = jnp.log(10000) / (half_dim - 1)
        embedding = jnp.exp(-embedding * jnp.arange(half_dim))
        embedding = time[:, None] * embedding[None, :]
        return jnp.concatenate([jnp.sin(embedding), jnp.cos(embedding)], axis=-1)
        
class ResBlock(nnx.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, rngs):
        self.conv1 = nnx.Conv(in_dim, out_dim, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.conv2 = nnx.Conv(out_dim, out_dim, kernel_size=(3, 3), padding=1, rngs=rngs)
        
        self.time_mlp = nnx.Linear(time_emb_dim, out_dim, rngs=rngs)
        
        self.norm1 = nnx.GroupNorm(num_features=in_dim, num_groups=4, rngs=rngs)
        self.norm2 = nnx.GroupNorm(num_features=out_dim, rngs=rngs)
        
        self.shortcut = nnx.Conv(in_dim, out_dim, kernel_size=(1, 1), rngs=rngs) if in_dim != out_dim else lambda x: x
    
    def __call__(self, x, t_emb):
        h = self.conv1(nnx.silu(self.norm1(x)))
        time_proj = self.time_mlp(nnx.silu(t_emb))
        
        h = h + time_proj[:, None, None, :]
        h = self.conv2(nnx.silu(self.norm2(h)))
        return h + self.shortcut(x)
    
class UNet(nnx.Module):
    def __init__(self, in_dim, base_dim, rngs):
        self.time_embed = SinusoidalEmbedding(base_dim)
        self.time_mlp = nnx.Linear(base_dim, base_dim*4, rngs=rngs)
        
        self.init_conv = nnx.Conv(in_dim, base_dim, kernel_size=(3, 3), padding=1, rngs=rngs)
        
        self.down1 = ResBlock(base_dim, base_dim, base_dim*4, rngs=rngs)
        self.pool = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.mid = ResBlock(base_dim, base_dim*2, base_dim*4, rngs=rngs)
        self.up_conv = nnx.ConvTranspose(base_dim*2, base_dim, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.up1 = ResBlock(base_dim*2, base_dim, base_dim*4, rngs=rngs)
        
        self.final_conv = nnx.Conv(base_dim, in_dim, kernel_size=(1,1), rngs=rngs)
        
    def __call__(self, x, time):
        time = self.time_mlp(self.time_embed(time))
        h = self.init_conv(x)
        d1 = self.down1(h, time)
        p = self.pool(d1)
        m = self.mid(p, time)
        u1 = self.up_conv(m)
        up_proj = self.up1(jnp.concatenate([u1, d1], axis=-1), time)
        
        return self.final_conv(up_proj)

def get_noise_schedule(T=1000):
    beta = jnp.linspace(1e-4, 0.02, T)
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    return alpha_bar

@nnx.jit
def train_step(model, optimizer, x, key, alpha_bar):
    B = x.shape[0]
    key_t, key_noise = jax.random.split(key)
    t = jax.random.randint(key_t, (B, ), minval=0, maxval=1000)
    
    a_bar_t = alpha_bar[t][:, None, None, None]
    noise = jax.random.normal(key_noise, x.shape)
    x_noisy = jnp.sqrt(a_bar_t) * x + jnp.sqrt(1 - a_bar_t) * noise
    
    def loss_fn(model):
        noise_pred = model(x_noisy, a_bar_t)
        return ((noise_pred - noise)**2).mean()
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

rngs = nnx.Rngs(0)
model = UNet(in_dim=3, base_dim=64, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

key = jax.random.key(42)
x_dummy = jax.random.normal(key, (8, 32, 32, 3))

alpha_bar = get_noise_schedule(T=1000)

for i in range(10):
    key, subkey = jax.random.split(key)
    loss = train_step(model, optimizer, x_dummy, subkey, alpha_bar)
    print(f"Step: {i} | Loss: {loss:.4f}")