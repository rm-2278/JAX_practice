from flax import linen as nn
import jax
import jax.numpy as jnp


class SimpleClassifier(nn.Module):
    num_hiddens : int
    num_outputs : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_hiddens)(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.num_outputs)(x)
        return x


model = SimpleClassifier(6, 1)

    
rng = jax.random.key(0)
key1, key2 = jax.random.split(rng, 2)
inp = jax.random.uniform(key1, (8, 2))

params = model.init(key2, inp)

print(model.apply(params, inp))