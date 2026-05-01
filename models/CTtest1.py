import jax
import jax.numpy as jnp
import optax
from flax import nnx
import math

class CausalSelfAttention(nnx.Module):
    def __init__(self, embed_dim, num_heads, rngs):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be exactly divisible by num_heads"
        
        self.q_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
                
    def __call__(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask==0, -1e9, scores)
        
        weights = nnx.softmax(scores, axis=-1)
        out = jnp.matmul(weights, v) #(B, H, T, d)
        
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)

class MLP(nnx.Module):
    def __init__(self, embed_dim, rngs):
        self.fc1 = nnx.Linear(embed_dim, 4*embed_dim, rngs=rngs)
        self.fc2 = nnx.Linear(embed_dim*4, embed_dim, rngs=rngs)
    
    def __call__(self, x):
        return self.fc2(nnx.gelu(self.fc1(x)))
    
    
class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, rngs):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn1 = CausalSelfAttention(embed_dim, num_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = MLP(embed_dim, rngs=rngs)
    
    def __call__(self, x):
        x = x + self.attn1(self.ln1(x))
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
        return self.lm_head(x)  # B, T, vocab_size
    
@nnx.jit
def train_step(model, optimizer, tokens):
    def loss_fn(model):
        logits = model(tokens)
        
        shift_logits = logits[:, :-1, :]
        shift_labels = tokens[:, 1:]
        
        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()
        return loss       
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grad)
    return loss
    
rngs = nnx.Rngs(0)
VOCAB_SIZE = 1000
EMBED_SIZE = 128
NUM_HEAD = 4
model = MiniGPT(vocab_size=VOCAB_SIZE, embed_dim=EMBED_SIZE, num_heads=NUM_HEAD, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

key = jax.random.key(1)
tokens = jax.random.randint(key, (16, 32), minval=0, maxval=VOCAB_SIZE)

for i in range(10):
    loss = train_step(model, optimizer, tokens)
    print(f"Step: {i} | Loss: {loss:.4f}")