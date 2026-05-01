import jax
import jax.numpy as jnp
import optax
from flax import nnx
import math

class CausalSelfAttention(nnx.Module):
    def __init__(self, embed_dim, num_heads, rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Q, K, V linear projections
        self.q_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x):
        B, T, C = x.shape
        
        # 1. Project and reshape into (Batch, Heads, Sequence, Head_Dim)
        # transpose(0, 2, 1, 3) moves the 'Heads' dimension before 'Sequence'
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 2. Compute Attention scores: Q @ K^T / sqrt(d)
        # Transpose K's last two dimensions: (B, H, d, T)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # 3. Causal Masking (Prevent looking into the future)
        mask = jnp.tril(jnp.ones((T, T)))
        # Replace 0s in the mask with -1e9 so they become 0 after softmax
        scores = jnp.where(mask == 0, -1e9, scores) 
        
        # 4. Softmax and multiply by V
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(weights, v) # Shape: (B, H, T, d)
        
        # 5. Reshape back to (Batch, Sequence, Embed_Dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)

class MLP(nnx.Module):
    def __init__(self, embed_dim, rngs):
        self.fc1 = nnx.Linear(embed_dim, 4 * embed_dim, rngs=rngs)
        self.fc2 = nnx.Linear(4 * embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x):
        return self.fc2(jax.nn.gelu(self.fc1(x)))

class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, rngs):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = CausalSelfAttention(embed_dim, num_heads, rngs)
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = MLP(embed_dim, rngs=rngs)

    def __call__(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nnx.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.block = TransformerBlock(embed_dim, num_heads, rngs)
        self.ln_f = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.lm_head = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def __call__(self, tokens):
        x = self.token_emb(tokens)
        x = self.block(x)
        x = self.ln_f(x)
        return self.lm_head(x) # Shape: (Batch, Sequence, Vocab_Size)

@nnx.jit
def train_step(model, optimizer, tokens):
    def loss_fn(model):
        logits = model(tokens)
        
        # Next-token prediction setup
        # Shift logits and targets so token N predicts token N+1
        shift_logits = logits[:, :-1, :]
        shift_labels = tokens[:, 1:]
        
        # Standard Cross Entropy Loss
        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()
        return loss

    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss

# --- Setup and Training Loop ---

rngs = nnx.Rngs(0)
VOCAB_SIZE = 1000
EMBED_DIM = 128
NUM_HEADS = 4

model = MiniGPT(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

# Dummy integer tokens: (Batch=16, Sequence=32)
key = jax.random.key(1)
tokens = jax.random.randint(key, (16, 32), minval=0, maxval=VOCAB_SIZE)

for i in range(10):
    loss = train_step(model, optimizer, tokens)
    print(f"Step {i} | Next-Token Prediction Loss: {loss:.4f}")