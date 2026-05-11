import jax
import jax.numpy as jnp
from flax import nnx
import optax

class SinusoidalEmbeddings(nnx.Module):
    def __init__(self, dim):
        self.dim = dim
    
    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(-embeddings * jnp.arange(half_dim))
        embeddings = time[:, None] * embeddings[None, :]
        return jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1) # Simply block concat since the learnable weight will allow flexibility
    
# 1. The Patchifier (Image -> Tokens)
class PatchEmbed(nnx.Module):
    def __init__(self, patch_size, in_ch, embed_dim, rngs):
        self.patch_size = patch_size
        # A Conv layer with kernel_size == stride perfectly slices the image into non-overlapping patches
        self.proj = nnx.Conv(in_ch, embed_dim, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size), rngs=rngs)

    def __call__(self, x):
        # x shape: (Batch, H, W, Channels)
        x = self.proj(x) 
        
        # New shape: (Batch, H/P, W/P, embed_dim)
        B, H, W, C = x.shape
        
        # Flatten the spatial dimensions to create a sequence: (Batch, Num_Patches, embed_dim)
        return x.reshape(B, H * W, C)

# 2. The DiT Block with adaLN-Zero
class DiTBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, rngs):
        # standard LayerNorm without learnable affine weights (we will inject them dynamically)
        self.norm1 = nnx.LayerNorm(num_features=embed_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, decode=False, rngs=rngs)
        
        self.norm2 = nnx.LayerNorm(num_features=embed_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim * 4, rngs=rngs),
            lambda x: jax.nn.gelu(x),
            nnx.Linear(embed_dim * 4, embed_dim, rngs=rngs)
        )
        
        # adaLN modulation: Predicts 6 parameters from the timestep
        # (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nnx.Sequential(
            lambda x: jax.nn.silu(x),
            nnx.Linear(embed_dim, 6 * embed_dim, rngs=rngs)
        )

    def __call__(self, x, c):
        # x: (Batch, Seq_Len, embed_dim) - The image tokens
        # c: (Batch, embed_dim) - The conditioned time embedding
        
        # 1. Predict LayerNorm parameters and skip connection gates from time
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(self.adaLN_modulation(c), 6, axis=-1)
        
        # Broadcast the 1D time parameters across the sequence length
        shift_msa, scale_msa, gate_msa = shift_msa[:, None, :], scale_msa[:, None, :], gate_msa[:, None, :]
        shift_mlp, scale_mlp, gate_mlp = shift_mlp[:, None, :], scale_mlp[:, None, :], gate_mlp[:, None, :]

        # 2. Attention Block + adaLN + Gated Residual
        normed_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(normed_x)

        # 3. MLP Block + adaLN + Gated Residual
        normed_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(normed_x)
        
        return x

# 3. The Full Diffusion Transformer
class DiT(nnx.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim, depth, num_heads, rngs):
        self.patch_embed = PatchEmbed(patch_size, in_ch, embed_dim, rngs)
        num_patches = (img_size // patch_size) ** 2
        
        # Positional Embedding (Just like your first NLP Transformer)
        self.pos_embed = nnx.Param(jax.random.normal(rngs(), (1, num_patches, embed_dim)) * 0.02)
        
        # Time Embedding
        self.t_embedder = SinusoidalEmbeddings(embed_dim)
        self.t_mlp = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
            lambda x: jax.nn.silu(x),
            nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        )
        
        # Stack of DiT Blocks
        self.blocks = [DiTBlock(embed_dim, num_heads, rngs) for _ in range(depth)]
        
        # Final Un-Patchify layer to get pixels back
        self.final_norm = nnx.LayerNorm(embed_dim, use_bias=False, use_scale=False, rngs=rngs)
        self.final_adaLN = nnx.Linear(embed_dim, 2 * embed_dim, rngs=rngs)
        
        # Maps the embedding dimension back to a block of raw pixels (patch_size * patch_size * in_ch)
        self.unpatchify = nnx.Linear(embed_dim, patch_size * patch_size * in_ch, rngs=rngs)

    def __call__(self, x, t):
        # 1. Process Time
        c = self.t_mlp(self.t_embedder(t))
        
        # 2. Patchify Image & Add Positional Information
        x = self.patch_embed(x) + self.pos_embed.value
        
        # 3. Forward through the Transformer Blocks
        for block in self.blocks:
            x = block(x, c)
            
        # 4. Final Output Prediction
        shift, scale = jnp.split(self.final_adaLN(jax.nn.silu(c)), 2, axis=-1)
        x = self.final_norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        
        # Predict the noise for each patch
        out = self.unpatchify(x)
        return out

# 1. The Patchify Helper
def patchify(imgs, patch_size):
    """
    Converts an image (B, H, W, C) into a sequence of patches (B, Seq_Len, Patch_Dim).
    """
    B, H, W, C = imgs.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Reshape into a grid of patches
    x = imgs.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C)
    # Swap axes to group the patches together: (B, num_patches_h, num_patches_w, patch_size, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    # Flatten the spatial grid into a sequence, and the patch pixels into a single vector
    x = x.reshape(B, num_patches_h * num_patches_w, patch_size * patch_size * C)
    return x

# 2. The Training Step
@nnx.jit(static_argnames=("patch_size"))
def train_step(model, optimizer, batch, alpha_bar, key, patch_size):
    # Split the key for timesteps and noise
    key_t, key_noise = jax.random.split(key)
    B = batch.shape[0]
    
    # Sample random timesteps for each image in the batch
    t = jax.random.randint(key_t, shape=(B,), minval=0, maxval=1000)
    
    # Generate the true noise
    noise = jax.random.normal(key_noise, batch.shape)
    
    # Gather the alpha_bar values for our specific timesteps
    a_bar_t = alpha_bar[t][:, None, None, None] # Reshape for broadcasting
    
    # Corrupt the images (The Forward Process)
    noisy_batch = jnp.sqrt(a_bar_t) * batch + jnp.sqrt(1 - a_bar_t) * noise
    
    # Patchify the true noise so it matches the DiT output shape
    target_noise_tokens = patchify(noise, patch_size)

    # Define the loss function for the optimizer to differentiate
    def loss_fn(model):
        # The DiT outputs the predicted noise tokens
        pred_noise_tokens = model(noisy_batch, t)
        # Calculate MSE in token space
        loss = jnp.mean((pred_noise_tokens - target_noise_tokens) ** 2)
        return loss

    # Calculate gradients and update the model weights
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    
    return loss

# 3. The Execution Loop
def train_dit():
    # Hyperparameters
    IMG_SIZE = 32
    PATCH_SIZE = 4
    IN_CH = 3
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8
    BATCH_SIZE = 16
    STEPS = 10
    
    # Initialize RNG
    rngs = nnx.Rngs(0)
    
    # Initialize the Model and Optimizer
    print("Initializing DiT...")
    model = DiT(IMG_SIZE, PATCH_SIZE, IN_CH, EMBED_DIM, DEPTH, NUM_HEADS, rngs)
    
    # We use AdamW because Transformers are highly prone to overfitting without weight decay
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-4, weight_decay=1e-4))
    
    # Pre-compute the alpha_bar schedule (Linear schedule for simplicity)
    beta = jnp.linspace(0.0001, 0.02, 1000)
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    
    # Create dummy image data [-1, 1]
    dummy_data = jax.random.uniform(rngs(), (BATCH_SIZE, IMG_SIZE, IMG_SIZE, IN_CH), minval=-1.0, maxval=1.0)
    
    print("\nStarting Training Loop...")
    print("-" * 30)
    
    # The Loop
    key = jax.random.PRNGKey(42)
    for step in range(STEPS):
        key, subkey = jax.random.split(key)
        
        # Execute the JIT-compiled train step
        loss = train_step(model, optimizer, dummy_data, alpha_bar, subkey, PATCH_SIZE)
        
        # Meaningful printing
        print(f"Step: {step:03d} | Token MSE Loss: {loss:.4f}")

if __name__ == "__main__":
    train_dit()