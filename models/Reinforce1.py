import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym

class Policy(nnx.Module):
    def __init__(self, obs_dim, action_dim, rngs):
        self.dense = nnx.Linear(obs_dim, 128, rngs=rngs)
        self.head = nnx.Linear(128, action_dim, rngs=rngs)
        
    def __call__(self, x):
        return self.head(nnx.relu(self.dense(x)))
    
@nnx.jit
def select_action(model, state, key):
    logits = model(state)
    action = jax.random.categorical(key, logits)
    return action

@nnx.jit
def compute_returns(rewards, gamma=0.99):
    def step(carry, r):
        g = r + gamma * carry
        return g, g
    _, returns = jax.lax.scan(step, 0.0, rewards[::-1])
    return returns[::-1]

def reinforce_loss(model, states, actions, returns):
    logits = model(states)
    log_probs = nnx.log_softmax(logits)
    log_probs_actions = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze()
    
    return -jnp.mean(returns * log_probs_actions)

@nnx.jit
def train_step(model, optimizer, states, actions, returns):
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    loss, grad = nnx.value_and_grad(reinforce_loss)(model, states, actions, returns_norm)
    optimizer.update(model, grad)
    return loss

def train():
    env = gym.make("CartPole-v1")
    rngs = nnx.Rngs(42)
    key = jax.random.key(42)
    
    model = Policy(obs_dim=4, action_dim=2, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    epochs = 500
    
    for epoch in range(1, epochs+1):
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        done = False
        
        while not done:
            key, subkey = jax.random.split(key)
            state_f = jnp.asarray(state, dtype=jnp.float32)
            
            action = select_action(model, state_f, subkey)
            action_int = int(action)
            
            next_state, reward, term, trunc, _ = env.step(action_int)
            done = term or trunc
            
            states.append(state_f)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        batch_states = jnp.stack(states)
        batch_actions = jnp.array(actions)
        batch_rewards = jnp.array(rewards, dtype=jnp.float32)
        
        returns = compute_returns(batch_rewards)
        
        loss = train_step(model, optimizer, batch_states, batch_actions, returns)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Eplength: {len(rewards):4d} | Loss: {float(loss):3.3f}")
        
train()
            
