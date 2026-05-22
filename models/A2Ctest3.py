import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from flax import nnx

class ContinuousActorCritic(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        self.dense1 = nnx.Linear(obs_dim, 128, rngs=rngs)
        self.mu_head = nnx.Linear(128, act_dim, rngs=rngs)
        
        self.dense2 = nnx.Linear(obs_dim, 128, rngs=rngs)
        self.value_head = nnx.Linear(128, 1, rngs=rngs)
        
        self.logstd = nnx.Param(jnp.zeros(act_dim))
        
    def __call__(self, x):
        mu = self.mu_head(nnx.relu(self.dense1(x)))
        value = self.value_head(nnx.relu(self.dense2(x))).squeeze()
        return mu, self.logstd.get_value(), value
    
        
@nnx.jit
def select_action(model, state, key):
    mu, logstd, value = model(state)
    action = mu + jnp.exp(logstd) * jax.random.normal(key, mu.shape)
    return action, value

@nnx.jit
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    values = jnp.append(values, next_value)
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    def step(carry, args):
        delta, done = args
        gae = delta + gamma * lam * carry * (1 - done)
        return gae, gae
    _, advantages = jax.lax.scan(step, 0.0, [deltas[::-1], dones[::-1]])
    advantages = advantages[::-1]
    return advantages, advantages + values[:-1]

def a2c_loss(model, states, actions, advantages, returns):
    mu, logstd, values = model(states)
    var = jnp.exp(2 * logstd)
    
    logprob = -0.5 * ((actions - mu)**2 / var + jnp.log(2 * jnp.pi) + 2 * logstd)
    logprobs = jnp.sum(logprob, axis=-1)
    
    pi_loss = -jnp.mean(logprobs * advantages)

    v_loss = 0.5 * jnp.mean((values - returns)**2)

    entropy = 0.5 * jnp.mean(jnp.sum(1.0 + jnp.log(2 * jnp.pi) + 2 * logstd, axis=-1))
    
    return pi_loss + v_loss - 0.01 * entropy
    

@nnx.jit
def train_step(model, optimizer, states, actions, advantages, returns):
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss, grads = nnx.value_and_grad(a2c_loss)(model, states, actions, adv_norm, returns)
    optimizer.update(model, grads)
    return loss
    


env = gym.make("Pendulum-v1")
rngs = nnx.Rngs(42)
key = jax.random.key(42)
# print(env.observation_space.shape, env.action_space.shape)
model = ContinuousActorCritic(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0], rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

state, _ = env.reset()

epochs = 1000
steps_per_epoch = 200

for epoch in range(epochs):
    states, actions, rewards, values, dones = [], [], [], [], []
    ep_return = 0.0

    for _ in range(steps_per_epoch):
        key, subkey = jax.random.split(key)
        state_f = jnp.asarray(state, dtype=jnp.float32)
        action, value = select_action(model, state_f, subkey)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        states.append(state_f)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        
        state = next_state
        
        ep_return += reward
        if done:
            state, _ = env.reset()
        
    batch_states = jnp.stack(states)
    batch_actions = jnp.stack(actions)
    batch_rewards = jnp.array(rewards, dtype=jnp.float32)
    batch_values = jnp.array(values)
    batch_dones = jnp.array(dones, dtype=jnp.float32)
    
    _, _, next_value = model(state)
    advantages, returns = compute_gae(batch_rewards, batch_values, batch_dones, next_value)
    loss = train_step(model, optimizer, batch_states, batch_actions, advantages, returns)
    
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Return: {ep_return} | Loss: {loss}")