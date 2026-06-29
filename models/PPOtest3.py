import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from flax import nnx
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
def select_action(model: PPO, state: jnp.array, key):
    mu, logstd, value = model(state)
    std = jnp.exp(logstd)
    action = mu + std * jax.random.normal(key, mu.shape)
    log_prob = -0.5 * (((action - mu) / std)**2 + jnp.log(2.0 * jnp.pi) + 2.0 * logstd)
    return action, jnp.sum(log_prob, axis=-1), value

@nnx.jit
def compute_gae(rewards, values, terminations, truncations, next_values, gamma=0.99, lam=0.95):
    deltas = rewards + gamma * next_values * (1.0 - terminations) - values
    def step(carry, args):
        episode_end, delta = args
        adv = delta + carry * gamma * lam * (1.0 - episode_end)
        return adv, adv
    _, advantages = jax.lax.scan(step, 0.0, [jnp.logical_or(terminations, truncations)[::-1], deltas[::-1]])
    advantages = advantages[::-1]
    return advantages, advantages + values

def ppo_loss(model: PPO, states, actions, old_log_probs, advantages, returns):
    mus, logstds, values = model(states)
    stds = jnp.exp(logstds)
    log_probs = -0.5 * jnp.sum(((actions - mus) / stds)**2 + jnp.log(2.0 * jnp.pi) + 2.0 * logstds, axis=-1)
    
    ratio = jnp.exp(log_probs - old_log_probs)
    pi_loss = -jnp.mean(jnp.minimum(ratio*advantages, jnp.clip(ratio, 0.8, 1.2) * advantages))
    v_loss = 0.5 * jnp.mean((values - returns)**2)
    entropy = jnp.mean(jnp.sum(0.5 * (1.0 + jnp.log(2.0 * jnp.pi) + logstds), axis=-1))
    
    return pi_loss + 0.5 * v_loss - 0.01 * entropy

@nnx.jit
def train_step(model, optimizer, states, actions, advantages, old_log_probs, returns):
    loss, grads = nnx.value_and_grad(ppo_loss)(model, states, actions, old_log_probs, advantages, returns)
    optimizer.update(model, grads)
    return loss

rngs= nnx.Rngs(42)
key = jax.random.key(42)
env = gym.make('Pendulum-v1')
model = PPO(obs_dim=3, act_dim=1, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

for epoch in range(100):
    state, _ = env.reset()
    states, actions, log_probs, rewards, values, next_states, terminations, truncations = [], [], [], [], [], [], [], []
    ep_return, eps, total_return = 0.0, 0.0, 0.0
    
    for _ in range(1000):
        key, subkey = jax.random.split(key)
        action, log_prob, value = select_action(model, jnp.array(state, dtype=jnp.float32), subkey)
        next_state, reward, term, trunc, _ = env.step(np.clip(action, -2.0, 2.0))
        
        states.append(state); actions.append(action); log_probs.append(log_prob)
        rewards.append(reward / 10.0); values.append(value)
        terminations.append(term); truncations.append(trunc); next_states.append(next_state)
        
        ep_return += reward
        state = next_state
        if term or trunc:
            state, _ = env.reset()
            total_return += ep_return
            ep_return = 0.0
            eps += 1
            
    next_states_arr = jnp.array(next_states, dtype=jnp.float32)
    _, _, next_values = model(next_states_arr)
    
    advantages, returns = compute_gae(
        jnp.array(rewards, dtype=jnp.float32),
        jnp.array(values, dtype=jnp.float32),
        jnp.array(terminations, dtype=jnp.bool_),
        jnp.array(truncations, dtype=jnp.bool_),
        next_values
    )
    
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    full_batch = {
        'states': jnp.stack(states, dtype=jnp.float32),
        'actions': jnp.array(actions, dtype=jnp.float32),
        'advantages': adv_norm,
        'old_log_probs': jnp.array(log_probs, dtype=jnp.float32),
        'returns': returns
    }
    
    for _ in range(10):
        indices = np.random.permutation(1000)
        for start in range(0, 1000, 250):
            mb_idx = indices[start:start+250]
            mb = {k: v[mb_idx] for k, v in full_batch.items()}
            loss = train_step(model, optimizer, mb['states'], mb['actions'], mb['advantages'], mb['old_log_probs'], mb['returns'])
        
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | AvgEpReturn: {(total_return / eps):.1f} | Loss: {float(loss):.3f}")
        