import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from flax.training import train_state, checkpoints

from model import VAE

def loss_fn(key, model, params, batch):
    imgs, _ = batch
    recon_logits, mu, logvar = model.apply({'params': params}, key, imgs)
    # loss is negative ELBO(L)
    # recon_imgs = jax.nn.sigmoid(recon_logits) # Already clipped
    # recon_loss = -jnp.sum(imgs * jnp.log(recon_imgs) + (1 - imgs)  * jnp.log(1 - recon_imgs), axis=(1, 2)).mean() # Binary cross entropy loss
    recon_loss = optax.sigmoid_binary_cross_entropy(recon_logits, imgs).sum(axis=(1, 2)).mean() # Numerically more stable, applies sigmoid
    kl_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1).mean()
    return recon_loss + kl_loss

class TrainerModule:
    def __init__(self, latent_dim, train_dataloader, val_dataloader, log_dir, seed=42):
        self.latent_dim = latent_dim
        self.seed = seed
        self.init_key = jax.random.key(self.seed)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.example_img = next(iter(self.train_dataloader))[0][:8]
        self.model = VAE(latent_dim)
        self.logger = SummaryWriter()
        
        self.log_dir=os.path.abspath( # orbax tensorstore refuse relative path
            os.path.join(log_dir, f'mnist_{latent_dim}')
        )
        
        self.create_functions()
        
        self.init_model()
        
    def create_functions(self):
        def train_step(key, batch, state):
            loss, grads = jax.value_and_grad(loss_fn, argnums=2)(key, self.model, state.params, batch)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        def eval_step(key, batch, state):
            return loss_fn(key, self.model, state.params, batch)
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)
    
    def init_model(self):
        init_key, call_key = jax.random.split(self.init_key)
        params = self.model.init(init_key, call_key, self.example_img)['params']
        optimizer = optax.adagrad(1e-3)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)


    def train_model(self, key, num_epochs=1500): # Originally 1500
        best_loss = 1e6
        for epoch_idx in tqdm(range(num_epochs)):
            key, subkey = jax.random.split(key)
            self.train_epoch(subkey, epoch_idx)
            if epoch_idx % 50 == 0: # Reduced logging
                key, eval_key = jax.random.split(key)
                eval_loss = self.eval_model(eval_key, self.val_dataloader)
                self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_model(step=epoch_idx)
                self.logger.flush()
            
    def train_epoch(self, key, epoch):
        losses = []
        for batch in self.train_dataloader:
            key, subkey = jax.random.split(key)
            self.state, loss = self.train_step(subkey, batch, self.state)
            losses.append(loss)
        losses = np.stack(jax.device_get(losses))
        avg_loss = losses.mean()
        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
    
    def eval_model(self, key, val_dataloader):
        losses = []
        batch_sizes = []
        for batch in val_dataloader:
            key, subkey = jax.random.split(key)
            loss = self.eval_step(subkey, batch, self.state)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses = np.stack(jax.device_get(losses))
        batch_sizes = np.stack(batch_sizes)
        avg_loss = (losses * batch_sizes).sum() / batch_sizes.sum()
        return avg_loss
            
    def save_model(self, step=0):
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, step=step, target=self.state.params, prefix=f'mnist_{self.latent_dim}')
        
    def load_model(self):
        params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f'mnist_{self.latent_dim}')
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)
        
    def checkpoint_exists(self):
        return checkpoints.latest_checkpoint(ckpt_dir=self.log_dir, prefix=f'mnist_{self.latent_dim}') is not None