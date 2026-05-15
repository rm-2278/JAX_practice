import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym

# 1. The Policy Network (Actor-Only)
class Policy(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.dense = nnx.Linear(obs_dim, 128, rngs=rngs)
        self.head = nnx.Linear(128, action_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = jax.nn.relu(self.dense(x))
        return self.head(x)

# 2. Fast Rollout Inference
@nnx.jit
def select_action(model: Policy, state: jnp.ndarray, key: jax.Array):
    logits = model(state)
    action = jax.random.categorical(key, logits)
    return action

# 3. Monte Carlo Returns (Dynamic Programming via JAX Scan)
@jax.jit
def compute_returns(rewards: jnp.ndarray, gamma: float = 0.99):
    def step(carry, r):
        # G_t = r_t + gamma * G_{t+1}
        g = r + gamma * carry
        return g, g # Carry, output
    # Scan backward through the episode rewards
    _, returns = jax.lax.scan(step, 0.0, rewards[::-1]) # f, init, xs -> carry, stacked output
    return returns[::-1]

# 4. The Policy Gradient Objective (The Log-Derivative Trick)
def reinforce_loss(model: Policy, states: jnp.ndarray, actions: jnp.ndarray, returns: jnp.ndarray):
    logits = model(states)
    log_probs = jax.nn.log_softmax(logits)
    log_probs_act = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze()
    
    # Loss = - E[ G_t * log(pi(a|s)) ]
    return -jnp.mean(returns * log_probs_act)

# 5. Optimization Step
@nnx.jit
def train_step(model: Policy, optimizer: nnx.Optimizer, 
               states: jnp.ndarray, actions: jnp.ndarray, returns: jnp.ndarray):
               
    # Zero-mean variance normalization: The simplest "baseline" to stabilize the gradient
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss, grads = nnx.value_and_grad(reinforce_loss)(model, states, actions, returns_norm)
    optimizer.update(model, grads)
    return loss

# 6. Minimal Execution Loop
def train():
    env = gym.make("CartPole-v1")
    rngs = nnx.Rngs(42)
    prng_key = jax.random.PRNGKey(42)
    
    model = Policy(obs_dim=4, action_dim=2, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    epochs = 500
    
    for epoch in range(epochs):
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        done = False
        
        # --- PHASE 1: GENERATE A SINGLE TRAJECTORY ---
        while not done:
            prng_key, subkey = jax.random.split(prng_key)
            state_f = jnp.asarray(state, dtype=jnp.float32)
            
            action = select_action(model, state_f, subkey)
            action_int = int(action)
            
            next_state, reward, term, trunc, _ = env.step(action_int)
            done = term or trunc
            
            states.append(state_f)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
        # Convert to JAX arrays
        batch_states = jnp.stack(states)
        batch_actions = jnp.array(actions)
        batch_rewards = jnp.array(rewards, dtype=jnp.float32)
        
        # --- PHASE 2: CALCULATE MONTE CARLO RETURNS ---
        returns = compute_returns(batch_rewards)
        
        # --- PHASE 3: OPTIMIZE (Single Step) ---
        # Unlike PPO, REINFORCE cannot reuse data. One trajectory = One step.
        loss = train_step(model, optimizer, batch_states, batch_actions, returns)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | EpLength: {len(rewards):4d} | Loss: {float(loss):6.3f}")

if __name__ == "__main__":
    train()