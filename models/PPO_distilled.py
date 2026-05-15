import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym

# 1. The Actor-Critic Architecture
class ActorCritic(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.common = nnx.Linear(obs_dim, 64, rngs=rngs)
        self.actor = nnx.Linear(64, action_dim, rngs=rngs)
        self.critic = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = jax.nn.relu(self.common(x))
        return self.actor(x), self.critic(x).squeeze()

# 2. Fast Rollout Inference
@nnx.jit
def get_action_and_value(model: ActorCritic, state: jnp.ndarray, key: jax.Array):
    logits, value = model(state)
    action = jax.random.categorical(key, logits)
    log_prob = jax.nn.log_softmax(logits)[action]
    return action, log_prob, value

# 3. Generalized Advantage Estimation (GAE)
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

# 4. PPO Loss & Metrics
def ppo_loss(model: ActorCritic, states: jnp.ndarray, actions: jnp.ndarray, 
             log_probs_old: jnp.ndarray, advantages: jnp.ndarray, returns: jnp.ndarray, 
             clip_eps: float = 0.2):
    
    logits, values = model(states)
    
    log_probs = jax.nn.log_softmax(logits)
    log_probs_act = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze()
    ratio = jnp.exp(log_probs_act - log_probs_old)
    
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pi_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    v_loss = 0.5 * jnp.mean((values - returns) ** 2)
    
    probs = jax.nn.softmax(logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
    
    approx_kl = jnp.mean(log_probs_old - log_probs_act)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
    
    total_loss = pi_loss + v_loss - 0.01 * entropy
    
    metrics = {
        "pi_loss": pi_loss, "v_loss": v_loss, "entropy": entropy,
        "approx_kl": approx_kl, "clip_frac": clip_frac
    }
    return total_loss, metrics

# 5. Optimization Step
@nnx.jit
def train_step(model: ActorCritic, optimizer: nnx.Optimizer, 
               states: jnp.ndarray, actions: jnp.ndarray, log_probs_old: jnp.ndarray, 
               advantages: jnp.ndarray, returns: jnp.ndarray):
    
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    grad_fn = nnx.value_and_grad(ppo_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(model, states, actions, log_probs_old, adv_norm, returns)
    
    optimizer.update(model, grads)
    return loss, metrics

# 6. Execution Loop
def train():
    env = gym.make("CartPole-v1")
    rngs = nnx.Rngs(42)
    prng_key = jax.random.PRNGKey(42)
    
    model = ActorCritic(obs_dim=4, action_dim=2, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    steps_per_epoch = 512
    epochs = 150
    
    for epoch in range(epochs):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        ep_return, ep_len = 0.0, 0
        episode_returns, episode_lengths = [], []
        
        state, _ = env.reset()
        
        # --- PHASE 1: TRAJECTORY COLLECTION ---
        for _ in range(steps_per_epoch):
            prng_key, subkey = jax.random.split(prng_key)
            state_f = jnp.asarray(state, dtype=jnp.float32)
            
            # Fast compiled inference
            action, log_prob, value = get_action_and_value(model, state_f, subkey)
            action_int = int(action)
            
            next_state, reward, term, trunc, _ = env.step(action_int)
            done = term or trunc
            
            states.append(state_f)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            ep_return += reward
            ep_len += 1
            
            if done:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_len)
                state, _ = env.reset()
                ep_return, ep_len = 0.0, 0
            else:
                state = next_state
        
        # Bootstrap final value for GAE
        _, _, next_value = get_action_and_value(model, jnp.asarray(state, dtype=jnp.float32), prng_key)
        
        # Convert buffer to device arrays
        batch_states = jnp.stack(states)
        batch_actions = jnp.array(actions)
        batch_log_probs = jnp.array(log_probs)
        batch_rewards = jnp.array(rewards, dtype=jnp.float32)
        batch_values = jnp.array(values)
        batch_dones = jnp.array(dones, dtype=jnp.float32)
        
        # --- PHASE 2: GAE & TARGETS ---
        advantages, returns = compute_gae(batch_rewards, batch_values, batch_dones, next_value)
        
        # --- PHASE 3: OPTIMIZE (Single mini-epoch for simplicity) ---
        loss, metrics = train_step(model, optimizer, batch_states, batch_actions, 
                                   batch_log_probs, advantages, returns)
        
        # --- PHASE 4: TELEMETRY ---
        if epoch % 10 == 0:
            avg_ret = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
            print(f"Epoch {epoch:03d} | AvgReturn: {avg_ret:6.1f} | "
                  f"Loss: {float(loss):6.3f} | PiLoss: {float(metrics['pi_loss']):6.3f} | "
                  f"VLoss: {float(metrics['v_loss']):6.3f} | KL: {float(metrics['approx_kl']):6.4f}")

if __name__ == "__main__":
    train()