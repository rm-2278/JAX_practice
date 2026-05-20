import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np

class ContinuousActorCritic(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.shared = nnx.Linear(obs_dim, 128, rngs=rngs)
        self.mu_head = nnx.Linear(128, act_dim, rngs=rngs)
        self.value_head = nnx.Linear(128, 1, rngs=rngs)
        
        self.log_std = nnx.Param(jnp.zeros(act_dim))
        
    def __call__(self, x):
        x = nnx.relu(self.shared(x))
        mu = self.mu_head(x)
        value = self.value_head(x).squeeze()
        return mu, self.log_std.get_value(), value

@nnx.jit
def select_action(model, state, key):
    mu, log_std, value = model(state)
    std = jnp.exp(log_std)
    
    action = mu + std * jax.random.normal(key, shape=mu.shape)
    return action, value

@nnx.jit
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    v_ext = jnp.append(values, next_value)
    deltas = rewards + gamma * v_ext[1:] * (1.0 - dones) - v_ext[:-1]
    
    def step(carry, args):
        delta, done = args
        gae = delta + gamma * lam * (1.0 - done) * carry
        return gae, gae
    
    _, advantages = jax.lax.scan(step, 0.0, (deltas[::-1], dones[::-1]))
    advantages = advantages[::-1]
    return advantages, advantages + values

def a2c_loss(model, states, actions, advantages, returns):
    mu, log_std, values = model(states)
    variance = jnp.exp(2 * log_std)
    
    log_prob = -0.5 * (((actions - mu)**2) / variance + jnp.log(variance) + jnp.log(2 * jnp.pi))
    log_probs = jnp.sum(log_prob, axis=-1)
    
    pi_loss = -jnp.mean(log_probs * advantages)
    
    v_loss = 0.5 * jnp.mean((values - returns) ** 2)
    
    entropy = 0.5 * jnp.mean(jnp.sum(1.0 + jnp.log(2 * jnp.pi) + jnp.log(variance), axis=-1))
    
    return pi_loss + v_loss - 0.01 * entropy

@nnx.jit
def train_step(model, optimizer, states, actions, advantages: jnp.ndarray, returns):
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss, grads = nnx.value_and_grad(a2c_loss)(model, states, actions, adv_norm, returns)
    optimizer.update(model, grads)
    return loss

env = gym.make("Pendulum-v1")
rngs = nnx.Rngs(42)
key = jax.random.key(42)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

model = ContinuousActorCritic(obs_dim, act_dim, rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

epochs = 2000
steps_per_epoch = 200

for epoch in range(epochs):
    state, _ = env.reset()
    states, actions, rewards, values, dones = [], [], [], [], []
    ep_return = 0.
    
    last_next_state = state
    last_done = False
    
    for _ in range(steps_per_epoch):
        key, subkey = jax.random.split(key)
        state_f = jnp.asarray(state, dtype=jnp.float32)
        
        action, value = select_action(model, state_f, subkey)
        env_action = np.clip(np.array(action), env.action_space.low, env.action_space.high)
        
        next_state, reward, term, trunc, _ = env.step(env_action)
        done = term or trunc
        
        last_next_state = next_state
        last_done = done
        
        states.append(state_f)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        
        ep_return += reward
        state = next_state
        if done:
            state, _ = env.reset()
            
    batch_states = jnp.stack(states)
    batch_actions = jnp.stack(actions)
    batch_rewards = jnp.array(rewards, dtype=jnp.float32)
    batch_values = jnp.array(values)
    batch_dones = jnp.array(dones, dtype=jnp.float32)
    
    if last_done:
        next_value = jnp.array(0.0, dtype=batch_values.dtype)
    else:
        _, _, next_value = model(jnp.asarray(state, dtype=jnp.float32))
    
    advantages, returns = compute_gae(batch_rewards, batch_values, batch_dones, next_value)
    
    loss = train_step(model, optimizer, batch_states, batch_actions, advantages, returns)
    
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Return: {ep_return:.1f} | Loss: {float(loss):.3f}")

    

    
    
        
    