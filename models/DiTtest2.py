import jax
import jax.numpy as jnp
from flax import nnx
import optax

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
        self.patch_size = patch_size
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
            lambda x: nnx.gelu(x),
            nnx.Linear(embed_dim*4, embed_dim, rngs=rngs)
        )
        
        self.adaLN_modulation = nnx.Sequential(
            lambda x: nnx.silu(x),
            nnx.Linear(embed_dim, embed_dim*6, rngs=rngs)
        )
    
    def __call__(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(self.adaLN_modulation(c), 6, axis=-1)
        shift_a, scale_a, gate_a = shift_a[:, None, :], scale_a[:, None, :], gate_a[:, None, :]
        shift_m, scale_m, gate_m = shift_m[:, None, :], scale_m[:, None, :], gate_m[:, None, :]
        
        normed_x = scale_a * (1 + self.norm1(x)) + shift_a
        x = x + self.attn(normed_x) * gate_a
        
        normed_x = scale_m * (1 + self.norm2(x)) + shift_m
        x = x + self.mlp(normed_x) * gate_m
        
        return x
    
class DiT(nnx.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim, num_heads, depth, rngs):
        self.patch_embed = PatchEmbed(patch_size, in_ch, embed_dim, rngs=rngs)
        num_patches = (img_size // patch_size)**2
        
        self.pos_embed = nnx.Param(jax.random.normal(rngs(), (1, num_patches, embed_dim)) * 0.02)
        
        self.t_embedder = SinusoidalEmbedding(embed_dim)
        self.t_mlp = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
            lambda x: nnx.silu(x),
            nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        )
        
        self.blocks = nnx.List([DiTBlock(embed_dim, num_heads, rngs=rngs) for i in range(depth)])
        
        self.final_norm = nnx.LayerNorm(embed_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.final_adaLN = nnx.Linear(embed_dim, embed_dim*2, rngs=rngs)
        
        self.unpatchify = nnx.Linear(embed_dim, patch_size*patch_size*in_ch, rngs=rngs)
        
    def __call__(self, x, c):
        c = self.t_mlp(self.t_embedder(c))
        x = self.patch_embed(x) + self.pos_embed.get_value()
        
        for block in self.blocks:
            x = block(x, c)
            
        shift, scale = jnp.split(self.final_adaLN(nnx.silu(c)), 2, axis=-1)
        x = self.final_norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        
        out = self.unpatchify(x)
        return out

def patchify(imgs, patch_size):
    B, H, W, C = imgs.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    x = imgs.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, num_patches_h*num_patches_w, patch_size*patch_size*C)
    return x

@nnx.jit(static_argnames=("patch_size"))
def train_step(model, optimizer, batch, alpha_bar, key, patch_size):
    key_t, key_noise = jax.random.split(key)
    
    t = jax.random.randint(key_t, (batch.shape[0],), minval=0, maxval=1000)
    
    a_bar_t = alpha_bar[t][:, None, None, None]
    noise = jax.random.normal(key_noise, batch.shape)
    target_noise = patchify(noise, patch_size)
    x_noisy = jnp.sqrt(a_bar_t)*batch + jnp.sqrt(1 - a_bar_t)*noise
    
    def loss_fn(model):
        pred_noise = model(x_noisy, t)
        return jnp.mean((pred_noise - target_noise)**2)
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

IMG_SIZE = 32
PATCH_SIZE = 4
IN_CH = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
BATCH_SIZE = 16

rngs = nnx.Rngs(0)
model = DiT(IMG_SIZE, PATCH_SIZE, IN_CH, EMBED_DIM, NUM_HEADS, DEPTH, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-4, weight_decay=1e-4), wrt=nnx.Param)

beta = jnp.linspace(1e-4, 0.02, 1000)
alpha = 1.0 - beta
alpha_bar = jnp.cumprod(alpha)

dummy_data = jax.random.uniform(rngs(), (BATCH_SIZE, IMG_SIZE, IMG_SIZE, IN_CH), minval=-1.0, maxval=1.0)

key = jax.random.key(42)
for i in range(10):
    key, subkey = jax.random.split(key)
    loss = train_step(model, optimizer, dummy_data, alpha_bar, subkey, patch_size=PATCH_SIZE)
    print(f"Step: {i} | MSE Token Loss: {loss:.4f}")