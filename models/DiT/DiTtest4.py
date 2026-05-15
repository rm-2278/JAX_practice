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
    
class PatchEmbed(nnx.Module):
    def __init__(self, patch_size, in_ch, embed_dim, rngs):
        self.proj = nnx.Conv(in_ch, embed_dim, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size), rngs=rngs)
        
    def __call__(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        return x.reshape(B, H*W, C)

class DiTBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, rngs):
        self.norm1 = nnx.LayerNorm(embed_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads, embed_dim, decode=False, rngs=rngs)
        
        self.norm2 = nnx.LayerNorm(embed_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim*4, rngs=rngs),
            nnx.gelu,
            nnx.Linear(embed_dim*4, embed_dim, rngs=rngs)
        )
        
        self.adaLNmodulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(embed_dim, embed_dim*6, rngs=rngs)
        )
        
    def __call__(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(self.adaLNmodulation(c), 6, axis=-1)
        shift_a, scale_a, gate_a = shift_a[:, None, :], scale_a[:, None, :], gate_a[:, None, :]
        shift_m, scale_m, gate_m = shift_m[:, None, :], scale_m[:, None, :], gate_m[:, None, :]
        
        normed_x = (1 + scale_a) * self.norm1(x) + shift_a
        x = x + self.attn(normed_x) * gate_a
        
        normed_x = (1 + scale_m) * self.norm2(x) + shift_m
        x = x + self.mlp(normed_x) * gate_m
        
        return x

class DiT(nnx.Module):
    def __init__(self, embed_dim, num_heads, patch_size, in_ch, img_size, depth, rngs):
        self.patch_embed = PatchEmbed(patch_size, in_ch, embed_dim, rngs=rngs)
        num_patch = (img_size // patch_size)**2
        self.pos_embed = nnx.Param(jax.random.normal(rngs(), (1, num_patch ,embed_dim)) * 0.02)
        
        self.t_embed = SinusoidalEmbedding(embed_dim)
        self.t_mlp = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        )
        
        self.blocks = nnx.List([DiTBlock(embed_dim, num_heads, rngs=rngs) for _ in range(depth)])
        
        self.final_norm = nnx.LayerNorm(embed_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.final_adaLN = nnx.Linear(embed_dim, embed_dim*2, rngs=rngs)
        
        self.unpatchify = nnx.Linear(embed_dim, patch_size*patch_size*in_ch, rngs=rngs)
        
    def __call__(self, x, c):
        t = self.t_mlp(self.t_embed(c))
        
        x = self.patch_embed(x) + self.pos_embed.get_value()
        
        for block in self.blocks:
            x = block(x, t)
        
        shift, scale = jnp.split(self.final_adaLN(nnx.silu(t)), 2, axis=-1)
        x = (1 + scale[:, None, :]) * self.final_norm(x) + shift[:, None, :]
        
        return self.unpatchify(x)
                
def patchify(imgs, patch_size):
    B, H, W, C = imgs.shape
    patch_h, patch_w = H // patch_size, W // patch_size
    x = imgs.reshape(B, patch_h, patch_size, patch_w, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, patch_h*patch_w, patch_size*patch_size*C)

@nnx.jit(static_argnames=("patch_size", ))
def train_step(model, optimizer, batch, alpha_bar, patch_size, key):
    key_t, key_noise = jax.random.split(key)
    
    t = jax.random.randint(key_t, (batch.shape[0],), minval=0, maxval=1000)
    
    noise = jax.random.normal(key_noise, batch.shape)
    target_noise = patchify(noise, patch_size)
    a_bar_t = alpha_bar[t][:, None, None, None]
    x_noisy = jnp.sqrt(a_bar_t) * batch + jnp.sqrt(1.0 - a_bar_t) * noise
    
    def loss_fn(model):
        pred_noise = model(x_noisy, t)
        return jnp.mean((pred_noise - target_noise)**2)
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss



rngs = nnx.Rngs(0)
model = DiT(embed_dim=256, num_heads=8, patch_size=4, in_ch=3, img_size=32, depth=6, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-4, weight_decay=1e-4), wrt=nnx.Param)
alpha = 1.0 - jnp.linspace(1e-4, 0.02, 1000)
alpha_bar = jnp.cumprod(alpha)

key = jax.random.key(42)
x_dummy = jax.random.uniform(rngs(), (16, 32, 32, 3), minval=-1.0, maxval=1.0)

for i in range(10):
    key, subkey = jax.random.split(key)
    loss = train_step(model, optimizer, x_dummy, alpha_bar, 4, subkey)
    print(f"Step: {i} | MSE token Loss: {loss:.4f}")