import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym


def discounted_returns(rewards: jnp.ndarray, dones: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Compute discounted returns with episode boundary handling.

    Args:
        rewards: shape (T,)
        dones: shape (T,), True when an episode ended at that timestep
        gamma: discount factor
    """
    rewards = rewards.astype(jnp.float32)
    dones = dones.astype(jnp.float32)

    def step(carry, x):
        r, d = x
        carry = r + gamma * carry * (1.0 - d)
        return carry, carry

    _, rets_rev = jax.lax.scan(step, 0.0, (rewards[::-1], dones[::-1]))
    return rets_rev[::-1]

# 1. The Actor-Critic Model
class ActorCritic(nnx.Module):
    def __init__(self, obs_dim, action_dim, rngs):
        # Shared feature extractor
        self.common = nnx.Linear(obs_dim, 64, rngs=rngs)
        # Actor head: Outputs logits for actions
        self.actor = nnx.Linear(64, action_dim, rngs=rngs)
        # Critic head: Outputs a single scalar value
        self.critic = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.common(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# 2. The PPO Loss Function
def ppo_loss(model, states, actions, log_probs_old, advantages, returns, clip_eps=0.2):
    logits, values = model(states)
    values = values.squeeze()
    
    # Policy Loss (Actor)
    log_probs = jax.nn.log_softmax(logits)
    # Get log_prob of the specific actions taken
    log_probs_act = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze()
    
    # ratio r_t(theta)
    ratio = jnp.exp(log_probs_act - log_probs_old)
    
    # Clipped objective
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    # Value Loss (Critic) - MSE between predicted value and actual returns
    value_loss = jnp.mean((values - returns) ** 2)
    
    # Entropy Loss (Encourages exploration) - reuse log_probs to avoid extra softmax
    probs = jnp.exp(log_probs)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
    
    return policy_loss + 0.5 * value_loss - 0.01 * entropy


def ppo_metrics(model, states, actions, log_probs_old, advantages, returns, clip_eps=0.2):
    """Compute PPO losses/diagnostics for logging (no gradients)."""
    logits, values = model(states)
    values = values.squeeze()

    log_probs = jax.nn.log_softmax(logits)
    log_probs_act = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze()

    ratio = jnp.exp(log_probs_act - log_probs_old)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    value_loss = jnp.mean((values - returns) ** 2)

    probs = jnp.exp(log_probs)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

    approx_kl = jnp.mean(log_probs_old - log_probs_act)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    total = policy_loss + 0.5 * value_loss - 0.01 * entropy
    return {
        "total": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }

# 3. Training Step
@nnx.jit
def train_step(model, optimizer, states, actions, log_probs_old, advantages, returns):
    def loss_fn(model):
        return ppo_loss(model, states, actions, log_probs_old, advantages, returns)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train_ppo():
    # 1. Setup
    env = gym.make("CartPole-v1")
    rngs = nnx.Rngs(42)
    model = ActorCritic(4, 2, rngs)
    optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)
    
    steps_per_epoch = 512  # Collect this many transitions before updating
    epochs = 100
    gamma = 0.99
    
    for epoch in range(epochs):
        # --- PHASE 1: COLLECTION (Interaction) ---
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        episode_returns = []
        episode_lengths = []
        ep_return = 0.0
        ep_len = 0
        state, _ = env.reset()
        
        for _ in range(steps_per_epoch):
            # Get action distribution and value estimate
            state_f = jnp.asarray(state, dtype=jnp.float32)
            logits, value = model(state_f)
            
            # Sample an action
            action = jax.random.categorical(rngs(), logits)
            log_prob = jax.nn.log_softmax(logits)[action]
            
            # Step the environment
            next_state, reward, term, trunc, _ = env.step(int(action))
            done = term or trunc

            ep_return += float(reward)
            ep_len += 1
            if done:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_len)
                ep_return = 0.0
                ep_len = 0
            
            # Store everything
            states.append(state_f)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(jnp.asarray(reward, dtype=jnp.float32))
            values.append(value.squeeze())
            dones.append(jnp.asarray(done, dtype=jnp.bool_))
            
            state = next_state if not done else env.reset()[0]

        # Convert lists to JAX arrays
        states = jnp.stack(states)
        actions, log_probs = jnp.array(actions), jnp.array(log_probs)
        rewards, values = jnp.array(rewards), jnp.array(values)
        dones = jnp.array(dones)

        # --- PHASE 2: CALCULATION (Advantages) ---
        # We calculate the "Advantage": How much better was this action than average?
        # A simple version: returns - baseline(values)
        returns = discounted_returns(rewards, dones, gamma)
        advantages = returns - values.astype(jnp.float32)
        # Normalize advantages (The "Secret Sauce" for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PHASE 3: OPTIMIZATION (The Step!) ---
        # We usually run multiple "mini-epochs" on the same batch of data
        loss = train_step(model, optimizer, states, actions, log_probs, advantages, returns)
        metrics = ppo_metrics(model, states, actions, log_probs, advantages, returns)
        
        if epoch % 10 == 0:
            if episode_returns:
                avg_ep_ret = sum(episode_returns) / len(episode_returns)
                avg_ep_len = sum(episode_lengths) / len(episode_lengths)
                last_ep_ret = episode_returns[-1]
                last_ep_len = episode_lengths[-1]
            else:
                avg_ep_ret = float("nan")
                avg_ep_len = float("nan")
                last_ep_ret = float("nan")
                last_ep_len = float("nan")
            print(
                f"Epoch {epoch} | Episodes: {len(episode_returns)} | "
                f"AvgEpReturn: {avg_ep_ret:.1f} | AvgEpLen: {avg_ep_len:.1f} | "
                f"LastEpReturn: {last_ep_ret:.1f} | LastEpLen: {last_ep_len:.1f} | "
                f"Loss: {float(loss):.4f} | "
                f"PiLoss: {float(metrics['policy_loss']):.3f} | VLoss: {float(metrics['value_loss']):.3f} | "
                f"Ent: {float(metrics['entropy']):.3f} | KL: {float(metrics['approx_kl']):.4f} | "
                f"ClipFrac: {float(metrics['clip_frac']):.3f}"
            )
    
if __name__ == "__main__":
    train_ppo()