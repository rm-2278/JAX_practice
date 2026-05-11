import jax
import jax.numpy as jnp
import optax
from flax import nnx

class SinusoidalEmbedding(nnx.Module):
    def __init__(self, dim):
        self.dim = dim
    
    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(-embeddings * jnp.arange(half_dim))
        embeddings = time[:, None] * embeddings[None, :]
        return jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1) # Simply block concat since the learnable weight will allow flexibility
    
class ResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.time_mlp = nnx.Linear(time_emb_dim, out_ch, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3, 3), padding=1, rngs=rngs)
        
        self.norm1 = nnx.GroupNorm(num_features=in_ch, num_groups=4, rngs=rngs)
        self.norm2 = nnx.GroupNorm(num_features=out_ch, num_groups=4, rngs=rngs)
        
        self.shortcut = nnx.Conv(in_ch, out_ch, kernel_size=(1, 1), rngs=rngs) if in_ch != out_ch else lambda x: x
        
    def __call__(self, x, t_emb):
        h = self.conv1(nnx.silu(self.norm1(x)))
        
        time_proj = self.time_mlp(nnx.silu(t_emb))
        h = h + time_proj[:, None, None, :]
        
        h = self.conv2(nnx.silu(self.norm2(h)))
        return h + self.shortcut(x)
        
class UNet(nnx.Module):
    def __init__(self, in_ch, base_ch, rngs):
        self.time_embed = SinusoidalEmbedding(base_ch)
        self.time_mlp = nnx.Linear(base_ch, base_ch*4, rngs=rngs)
        
        self.init_conv = nnx.Conv(in_ch, base_ch, kernel_size=(3, 3), padding=1, rngs=rngs)
        
        self.down1 = ResBlock(base_ch, base_ch, base_ch*4, rngs=rngs)
        self.pool = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        self.mid = ResBlock(base_ch, base_ch*2, base_ch*4, rngs=rngs)
        
        self.up_conv = nnx.ConvTranspose(base_ch*2, base_ch, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.up1 = ResBlock(base_ch*2, base_ch, base_ch*4, rngs=rngs)
        
        self.final_conv = nnx.Conv(base_ch, in_ch, kernel_size=(1, 1), rngs=rngs)
        
    def __call__(self, x, time):
        t_emb = self.time_embed(time)
        t_emb = self.time_mlp(t_emb)
        
        x = self.init_conv(x)
        d1 = self.down1(x, t_emb)
        p = self.pool(d1)
        m = self.mid(p, t_emb)
        u1 = self.up_conv(m)
        out = self.up1(jnp.concatenate([u1, d1], axis=-1), t_emb)
        
        return self.final_conv(out)
        

def generate_noise_schedule(T=1000):
    beta = jnp.linspace(1e-4, 0.02, T)
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    return alpha_bar
        
@nnx.jit
def train_step(model, optimizer, x, key, alpha_bar):
    B = x.shape[0]
    key_t, key_noise = jax.random.split(key)
    
    t = jax.random.randint(key_t, (B, ), minval=0, maxval=1000)
    
    noise = jax.random.normal(key_noise, x.shape)
    
    a_bar_t = alpha_bar[t][:, None, None, None]
    
    x_noisy = jnp.sqrt(a_bar_t) * x + jnp.sqrt(1 - a_bar_t) * noise
    
    def loss_fn(model):
        return ((model(x_noisy, t) - noise)**2).mean()
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

rngs = nnx.Rngs(0)
model = UNet(in_ch=3, base_ch=64, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

key = jax.random.key(42)
x_dummy = jax.random.normal(key, (8, 32, 32, 3))

alpha_bar = generate_noise_schedule(T=1000)

for i in range(10):
    key, subkey = jax.random.split(key)
    loss = train_step(model, optimizer, x_dummy, subkey, alpha_bar)
    print(f"Step: {i} | Loss: {loss:.4f}")