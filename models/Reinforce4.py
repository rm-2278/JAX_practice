import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym

class Policy(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.dense = nnx.Linear(obs_dim, 128, rngs=rngs)
        self.head = nnx.Linear(128, act_dim, rngs=rngs)
    def __call__(self, x):
        return self.head(nnx.relu(self.dense(x)))
    
@nnx.jit
def select_action(model, state, key):
    logits = model(state)
    action = jax.random.categorical(key, logits)
    return action

def compute_return(rewards, gamma=0.99):
    def step(carry, r):
        g = r + carry*gamma
        return g, g
    _, returns = jax.lax.scan(step, 0.0, rewards[::-1])
    return returns[::-1]

def reinfoce_loss(model, states, actions, returns):
    logits = model(states)
    log_probs = nnx.log_softmax(logits)
    log_probs_action = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
    return -jnp.mean(returns * log_probs_action)    # If log_probs_action is confidence, if high confidence in higher than average cases neg loss and push towards it, if high confidence in lower than average cases pos loss and push away from it
    
@nnx.jit
def train_step(model, optimizer, states, actions, returns):
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    loss, grad = nnx.value_and_grad(reinfoce_loss)(model, states, actions, returns_norm)
    optimizer.update(model, grad)
    return loss

SEED = 42
rngs = nnx.Rngs(SEED)
key = jax.random.key(SEED)

model = Policy(obs_dim=4, act_dim=2, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

env = gym.make('CartPole-v1')


for epoch in range(1, 501):
    state, _ = env.reset(seed=SEED+epoch)
    states, actions, rewards = [], [], []
    
    done = False
    
    while not done:
        key, subkey = jax.random.split(key)
        state_f = jnp.asarray(state, dtype=jnp.float32)
        action = select_action(model, state_f, subkey)
        next_state, reward, term, trunc, _ = env.step(int(action))
        done = term or trunc
        
        states.append(state_f)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        
    states = jnp.stack(states)
    actions = jnp.array(actions)
    returns = compute_return(jnp.array(rewards, dtype=jnp.float32))
    
    loss = train_step(model, optimizer, states, actions, returns)
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | EpLength: {len(returns)} | Loss: {loss:.4f}")
        