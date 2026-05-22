import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np

class PPO(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.actor = nnx.Sequential(nnx.Linear(obs_dim, 64, rngs=rngs), jax.nn.tanh, nnx.Linear(64, act_dim, rngs=rngs))
        self.critic = nnx.Sequential(nnx.Linear(obs_dim, 64, rngs=rngs), jax.nn.tanh, nnx.Linear(64, 1, rngs=rngs))
        self.log_std = nnx.Param(jnp.zeros(act_dim))

    def __call__(self, x):
        return self.actor(x), self.log_std.value, self.critic(x).squeeze(-1)

@nnx.jit
def select_action(model, state, key):
    mu, log_std, value = model(state)
    std = jnp.exp(log_std)
    action = mu + std * jax.random.normal(key, shape=mu.shape)
    log_prob = -0.5 * (((action - mu) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    return action, jnp.sum(log_prob, axis=-1), value

@jax.jit
def compute_gae(rewards, values, dones, next_val):
    v_ext = jnp.append(values, next_val)
    deltas = rewards + 0.99 * v_ext[1:] * (1.0 - dones) - v_ext[:-1]
    
    def step(gae, args):
        val = args[0] + 0.99 * 0.95 * (1.0 - args[1]) * gae
        return val, val

    _, adv = jax.lax.scan(step, 0.0, (deltas[::-1], dones[::-1]))
    return adv[::-1], adv[::-1] + values

def ppo_loss(model, states, actions, old_log_probs, advantages, returns):
    mu, log_std, values = model(states)
    std = jnp.exp(log_std)
    
    log_probs = jnp.sum(-0.5 * (((actions - mu) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)), axis=-1)
    ratio = jnp.exp(log_probs - old_log_probs)
    
    pi_loss = -jnp.mean(jnp.minimum(ratio * advantages, jnp.clip(ratio, 0.8, 1.2) * advantages))
    v_loss = 0.5 * jnp.mean((values - returns) ** 2)
    entropy = jnp.mean(jnp.sum(0.5 * (1.0 + jnp.log(2.0 * jnp.pi)) + log_std, axis=-1))
    
    return pi_loss + 0.5 * v_loss - 0.01 * entropy

@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(ppo_loss)(model, batch['states'], batch['actions'], batch['log_probs'], batch['advantages'], batch['returns'])
    optimizer.update(model, grads)
    return loss

def train():
    env = gym.make("Pendulum-v1")
    model = PPO(3, 1, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)
    prng_key = jax.random.PRNGKey(42)
    
    for epoch in range(1000):
        state, _ = env.reset()
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        ep_return, total_return, eps = 0, 0, 0
        
        for _ in range(1000):
            prng_key, subkey = jax.random.split(prng_key)
            action, log_prob, value = select_action(model, jnp.array(state, dtype=jnp.float32), subkey)
            
            env_action = np.clip(np.array(action), -2.0, 2.0)
            next_state, reward, term, trunc, _ = env.step(env_action)
            
            states.append(state); actions.append(action); log_probs.append(log_prob)
            
            # --- FIX 1: REWARD SCALING ---
            rewards.append(reward / 10.0) 
            values.append(value); dones.append(term)
            
            ep_return += reward  # Keep tracking the unscaled return for clean logging
            state = next_state
            if term or trunc:
                state, _ = env.reset()
                total_return += ep_return
                ep_return, eps = 0, eps + 1

        _, _, next_val = model(jnp.array(state, dtype=jnp.float32))
        advantages, returns = compute_gae(jnp.array(rewards), jnp.array(values), jnp.array(dones), float(next_val))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        full_batch = {'states': jnp.array(states), 'actions': jnp.array(actions), 'log_probs': jnp.array(log_probs), 'advantages': advantages, 'returns': returns}
        
        # --- FIX 2: MINIBATCHING ---
        for _ in range(10):
            indices = np.random.permutation(1000)
            for start in range(0, 1000, 250): # 4 minibatches of 250
                mb_idx = indices[start:start+250]
                mb = {k: v[mb_idx] for k, v in full_batch.items()}
                loss = train_step(model, optimizer, mb)
                
        if epoch % 10 == 0 and eps > 0:
            print(f"Epoch {epoch:02d} | Avg Return: {total_return/eps:7.1f} | Loss: {float(loss):6.3f}")

if __name__ == "__main__":
    train()