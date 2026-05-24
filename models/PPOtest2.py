import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np

class PPO(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.actor = nnx.Sequential(
            nnx.Linear(obs_dim, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, act_dim, rngs=rngs)
        )
        self.critic = nnx.Sequential(
            nnx.Linear(obs_dim, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 1, rngs=rngs)
        )
        self.logstd = nnx.Param(jnp.zeros(act_dim))
        
    def __call__(self, x):
        return self.actor(x), self.logstd.get_value(), self.critic(x).squeeze(-1)
    
@nnx.jit
def select_action(model, state, key):
    mu, logstd, value = model(state)
    std = jnp.exp(logstd)
    action = mu + std * jax.random.normal(key, mu.shape)
    log_prob = -0.5 * (((action - mu) / std)**2 + 2.0 * jnp.pi + 2.0 * jnp.pi + 2.0 * logstd)
    return action, jnp.sum(log_prob, axis=-1), value

@nnx.jit
def compute_gae(rewards, values, terminations, truncations, next_values, gamma=0.99, lam=0.95):
    deltas = rewards + gamma * values * terminations - next_values
    def step(carry, args):
        episode_end, delta = args
        adv = delta + carry * gamma * lam * (1.0 - episode_end)        
        return adv, adv
    _, advantages = jax.lax.scan(step, 0.0, [jnp.logical_or(terminations, truncations)[::-1], deltas[::-1]])
    return advantages, advantages + values

def ppo_loss():
    

@nnx.jit
def train_step():
    


env = gym.make("Pendulum-v1")
