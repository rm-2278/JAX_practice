import jax
import jax.numpy as jnp
import optax
from flax import nnx
import math

# 1. Sinusoidal Time Embedding (Like Positional Encoding, but for Time)
class SinusoidalTimeEmbeddings(nnx.Module):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, time):
        # time shape: (Batch,)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)

# 2. Residual Convolutional Block with Time Injection
class ResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.time_mlp = nnx.Linear(time_emb_dim, out_ch, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3, 3), padding=1, rngs=rngs)
        
        # GroupNorm is the standard for Diffusion Models, not BatchNorm
        self.norm1 = nnx.GroupNorm(num_groups=4, num_channels=in_ch, rngs=rngs)
        self.norm2 = nnx.GroupNorm(num_groups=4, num_channels=out_ch, rngs=rngs)
        
        # 1x1 conv to match channel dimensions for the residual addition if they differ
        self.shortcut = nnx.Conv(in_ch, out_ch, kernel_size=(1, 1), rngs=rngs) if in_ch != out_ch else lambda x: x

    def __call__(self, x, t_emb):
        # Image pathway
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        
        # Time injection: Project time to match channel dim, then add to image features
        # t_emb shape: (B, out_ch) -> Reshape to (B, 1, 1, out_ch) to broadcast across H, W
        time_proj = self.time_mlp(jax.nn.silu(t_emb))
        h = h + time_proj[:, None, None, :]
        
        h = self.conv2(jax.nn.silu(self.norm2(h)))
        return h + self.shortcut(x)

# 3. The U-Net
class UNet(nnx.Module):
    def __init__(self, in_ch, base_ch, rngs):
        self.time_embed = SinusoidalTimeEmbeddings(base_ch)
        self.time_mlp = nnx.Linear(base_ch, base_ch * 4, rngs=rngs)

        # Down Path
        self.down1 = ResBlock(in_ch, base_ch, base_ch * 4, rngs=rngs)
        self.pool = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Bottleneck (Middle)
        self.mid = ResBlock(base_ch, base_ch * 2, base_ch * 4, rngs=rngs)

        # Up Path
        self.up_conv = nnx.ConvTranspose(base_ch * 2, base_ch, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        # Note: in_ch is base_ch*2 because we concatenate the skip connection
        self.up1 = ResBlock(base_ch * 2, base_ch, base_ch * 4, rngs=rngs) 

        self.final_conv = nnx.Conv(base_ch, in_ch, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x, time):
        # 1. Process Time
        t_emb = self.time_embed(time)
        t_emb = self.time_mlp(t_emb)

        # 2. Down Path (Save activations for skip connection!)
        d1 = self.down1(x, t_emb)         # Shape: (B, H, W, base_ch)
        p1 = self.pool(d1)                # Shape: (B, H/2, W/2, base_ch)

        # 3. Middle Path
        m = self.mid(p1, t_emb)           # Shape: (B, H/2, W/2, base_ch*2)

        # 4. Up Path
        u1 = self.up_conv(m)              # Upsample back to (B, H, W, base_ch)
        
        # CRITICAL: The U-Net Skip Connection
        skip_cat = jnp.concatenate([u1, d1], axis=-1) # Shape: (B, H, W, base_ch*2)
        
        out = self.up1(skip_cat, t_emb)   # Shape: (B, H, W, base_ch)
        return self.final_conv(out)       # Shape: (B, H, W, in_ch)

# --- Setup and DDPM Training Loop ---

# Pre-compute the noise schedule (alpha_bar)
def get_noise_schedule(T=1000):
    beta = jnp.linspace(1e-4, 0.02, T)
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    return alpha_bar

@nnx.jit
def train_step(model, optimizer, x, key, alpha_bar):
    B = x.shape[0]
    key_t, key_noise = jax.random.split(key)
    
    # 1. Sample random timesteps for the batch
    t = jax.random.randint(key_t, (B,), minval=0, maxval=1000)
    
    # 2. Sample random Gaussian noise
    noise = jax.random.normal(key_noise, x.shape)
    
    # 3. Corrupt the images (Forward Process)
    # Extract alpha_bar for specific timesteps and reshape for broadcasting
    a_bar_t = alpha_bar[t][:, None, None, None]
    x_noisy = jnp.sqrt(a_bar_t) * x + jnp.sqrt(1 - a_bar_t) * noise

    # 4. Predict the noise and calculate MSE Loss
    def loss_fn(model):
        pred_noise = model(x_noisy, t)
        return ((pred_noise - noise)**2).mean()

    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

# Execution
rngs = nnx.Rngs(0)
model = UNet(in_ch=3, base_ch=64, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

alpha_bar = get_noise_schedule(T=1000)
key = jax.random.key(42)
# Dummy images: Batch=8, Height=32, Width=32, Channels=3 (e.g., CIFAR-10 size)
x_dummy = jax.random.normal(key, (8, 32, 32, 3))

for i in range(10):
    key, subkey = jax.random.split(key)
    loss = train_step(model, optimizer, x_dummy, subkey, alpha_bar)
    print(f"Step {i} | Noise Prediction Loss: {loss:.4f}")