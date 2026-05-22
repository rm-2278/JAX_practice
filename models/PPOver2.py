import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np

# 1. The Actor-Critic Architecture (Same as A2C)
class PPOActorCritic(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.shared = nnx.Linear(obs_dim, 64, rngs=rngs)
        self.mu_head = nnx.Linear(64, action_dim, rngs=rngs)
        self.value_head = nnx.Linear(64, 1, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x: jnp.ndarray):
        x = jax.nn.tanh(self.shared(x))
        mu = self.mu_head(x)
        value = self.value_head(x).squeeze()
        return mu, self.log_std.get_value(), value

# 2. Action and Log-Prob Extraction
@nnx.jit
def select_action(model: PPOActorCritic, state: jnp.ndarray, key: jax.Array):
    mu, log_std, value = model(state)
    std = jnp.exp(log_std)
    u = mu + std * jax.random.normal(key, shape=mu.shape)
    action = jnp.tanh(u)  # Squashed to env bounds
    
    # Tanh-squashed Gaussian Log-Probability calculation
    raw_log_prob = -0.5 * (((u - mu) ** 2) / (std ** 2) + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    jacobian_correction = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    log_prob = jnp.sum(raw_log_prob - jacobian_correction, axis=-1)
    
    return action, log_prob, value

# 3. GAE Target Generation (Same as A2C)
@jax.jit
def compute_gae(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray, next_value: float):
    gamma, lam = 0.99, 0.95
    v_ext = jnp.append(values, next_value)
    deltas = rewards + gamma * v_ext[1:] * (1.0 - dones) - v_ext[:-1]

    def step(carry, args):
        delta, done = args
        gae = delta + gamma * lam * (1.0 - done) * carry
        return gae, gae

    _, advantages = jax.lax.scan(step, 0.0, (deltas[::-1], dones[::-1]))
    return advantages[::-1], advantages[::-1] + values

# 4. The PPO Clipped Surrogate Objective
def ppo_loss(model: PPOActorCritic, states: jnp.ndarray, actions: jnp.ndarray, 
             old_log_probs: jnp.ndarray, advantages: jnp.ndarray, returns: jnp.ndarray, 
             clip_eps: float = 0.2):
    
    mu, log_std, values = model(states)
    std = jnp.exp(log_std)
    
    # Re-evaluate the fresh log-probabilities of the chosen actions
    u = jnp.arctanh(jnp.clip(actions, -0.999999, 0.999999))
    raw_log_prob = -0.5 * (((u - mu) ** 2) / (std ** 2) + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    jacobian_correction = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
    log_probs = jnp.sum(raw_log_prob - jacobian_correction, axis=-1)
    
    # Probability Ratio: r_t(theta)
    ratio = jnp.exp(log_probs - old_log_probs)
    
    # Clipped Policy Loss
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pi_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # Value Loss & Entropy
    v_loss = 0.5 * jnp.mean((values - returns) ** 2)
    entropy = 0.5 * jnp.mean(jnp.sum(1.0 + jnp.log(2 * jnp.pi) + log_std * 2.0, axis=-1))
    
    return pi_loss + 0.5 * v_loss - 0.01 * entropy

# 5. Multi-Epoch Training Step
@nnx.jit
def train_step(model: PPOActorCritic, optimizer: nnx.Optimizer, batch: dict):
    loss, grads = nnx.value_and_grad(ppo_loss)(
        model, batch['states'], batch['actions'], batch['log_probs'], batch['advantages'], batch['returns']
    )
    optimizer.update(model, grads)
    return loss

# 6. Environment Execution Loop
def train():
    env = gym.make("Pendulum-v1")
    rngs = nnx.Rngs(42)
    prng_key = jax.random.PRNGKey(42)
    
    model = PPOActorCritic(env.observation_space.shape[0], env.action_space.shape[0], rngs)
    optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)
    
    for epoch in range(1000):
        state, _ = env.reset()
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        ep_return = 0
        
        # Rollout Phase
        for _ in range(200):
            prng_key, subkey = jax.random.split(prng_key)
            state_f = jnp.asarray(state, dtype=jnp.float32)
            
            action, log_prob, value = select_action(model, state_f, subkey)
            next_state, reward, term, trunc, _ = env.step(np.array(action))
            
            states.append(state_f); actions.append(action); log_probs.append(log_prob)
            rewards.append(reward); values.append(value); dones.append(term or trunc)
            
            ep_return += reward
            state = next_state if not (term or trunc) else env.reset()[0]

        # Prepare Batch & Targets
        _, _, next_value = model(jnp.asarray(state, dtype=jnp.float32))
        advantages, returns = compute_gae(jnp.array(rewards), jnp.array(values), jnp.array(dones), float(next_value))
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch = {
            'states': jnp.stack(states), 'actions': jnp.stack(actions), 'log_probs': jnp.stack(log_probs),
            'advantages': adv_norm, 'returns': returns
        }
        
        # --- THE PPO DATA RECYCLING LOOP ---
        # We reuse the exact same trajectory data for 10 full epochs safely
        ppo_epochs = 10
        for _ in range(ppo_epochs):
            loss = train_step(model, optimizer, batch)
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch:02d} | Return: {ep_return:7.1f} | Loss: {float(loss):6.3f}")

if __name__ == "__main__":
    train()