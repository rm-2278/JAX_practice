import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np

# 1. The Continuous Actor-Critic Architecture
class ContinuousActorCritic(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.shared = nnx.Linear(obs_dim, 128, rngs=rngs)
        self.mu_head = nnx.Linear(128, action_dim, rngs=rngs)
        self.value_head = nnx.Linear(128, 1, rngs=rngs)
        
        # Standard deviation is typically a state-independent learnable parameter
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x: jnp.ndarray):
        x = jax.nn.relu(self.shared(x))
        mu = self.mu_head(x)
        value = self.value_head(x).squeeze()
        # Return mu, log_std, and value
        return mu, self.log_std.value, value

# 2. Action Selection via Reparameterization
@nnx.jit
def select_action(model: ContinuousActorCritic, state: jnp.ndarray, key: jax.Array):
    mu, log_std, value = model(state)
    std = jnp.exp(log_std)
    
    # a = mu + std * N(0, 1)
    action = mu + std * jax.random.normal(key, shape=mu.shape)
    return action, value

# 3. GAE (Retained to compute reliable Advantages)
@jax.jit
def compute_gae(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray, 
                next_value: float, gamma: float = 0.99, lam: float = 0.95):
    v_ext = jnp.append(values, next_value)
    deltas = rewards + gamma * v_ext[1:] * (1.0 - dones) - v_ext[:-1]

    def step(carry, args):
        delta, done = args
        gae = delta + gamma * lam * (1.0 - done) * carry
        return gae, gae

    _, advantages = jax.lax.scan(step, 0.0, (deltas[::-1], dones[::-1]))
    advantages = advantages[::-1]
    return advantages, advantages + values

# 4. Continuous A2C Loss Objective
def a2c_loss(model: ContinuousActorCritic, states: jnp.ndarray, actions: jnp.ndarray, 
             advantages: jnp.ndarray, returns: jnp.ndarray):
    
    mu, log_std, values = model(states)
    variance = jnp.exp(log_std) ** 2
    
    # Log-probability of a Multivariate Gaussian:
    # log pi(a|s) = -0.5 * [ ((a - mu)^2 / var) + log(var) + log(2*pi) ]
    log_prob = -0.5 * (((actions - mu) ** 2) / variance + jnp.log(variance) + jnp.log(2 * jnp.pi))
    log_probs = jnp.sum(log_prob, axis=-1)
    
    # A2C Policy Loss (Notice there is no clipping here, just the pure Advantage)
    pi_loss = -jnp.mean(log_probs * advantages)
    
    # Value Loss
    v_loss = 0.5 * jnp.mean((values - returns) ** 2)
    
    # Gaussian Entropy = 0.5 * (1 + log(2*pi) + log(var))
    entropy = 0.5 * jnp.mean(jnp.sum(1.0 + jnp.log(2 * jnp.pi) + jnp.log(variance), axis=-1))
    
    return pi_loss + v_loss - 0.01 * entropy

# 5. The Optimization Step
@nnx.jit
def train_step(model: ContinuousActorCritic, optimizer: nnx.Optimizer, 
               states: jnp.ndarray, actions: jnp.ndarray, 
               advantages: jnp.ndarray, returns: jnp.ndarray):
    
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss, grads = nnx.value_and_grad(a2c_loss)(model, states, actions, adv_norm, returns)
    optimizer.update(model, grads)
    return loss

# 6. Minimal Execution Loop for Continuous Control
def train():
    env = gym.make("Pendulum-v1")
    rngs = nnx.Rngs(42)
    prng_key = jax.random.PRNGKey(42)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    model = ContinuousActorCritic(obs_dim, action_dim, rngs)
    optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)
    
    epochs = 300
    steps_per_epoch = 200  # Pendulum truncates at 200 steps
    
    for epoch in range(epochs):
        state, _ = env.reset()
        states, actions, rewards, values, dones = [], [], [], [], []
        ep_return = 0.0
        
        for _ in range(steps_per_epoch):
            prng_key, subkey = jax.random.split(prng_key)
            state_f = jnp.asarray(state, dtype=jnp.float32)
            
            action, value = select_action(model, state_f, subkey)
            # Must clip the continuous action to the environment's legal bounds
            env_action = np.clip(np.array(action), env.action_space.low, env.action_space.high)
            
            next_state, reward, term, trunc, _ = env.step(env_action)
            done = term or trunc
            
            states.append(state_f)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            ep_return += reward
            state = next_state
            if done:
                state, _ = env.reset()
                
        # Device array conversion
        batch_states = jnp.stack(states)
        batch_actions = jnp.stack(actions)
        batch_rewards = jnp.array(rewards, dtype=jnp.float32)
        batch_values = jnp.array(values)
        batch_dones = jnp.array(dones, dtype=jnp.float32)
        
        # Bootstrap
        _, _, next_value = model(jnp.asarray(state, dtype=jnp.float32))
        
        advantages, returns = compute_gae(batch_rewards, batch_values, batch_dones, float(next_value))
        
        # A2C Update (Single step, no multiple epochs per rollout)
        loss = train_step(model, optimizer, batch_states, batch_actions, advantages, returns)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Return: {ep_return:7.1f} | Loss: {float(loss):6.3f}")

if __name__ == "__main__":
    train()