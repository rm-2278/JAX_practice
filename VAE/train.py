import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from flax.training import train_state

from model import VAE

def loss_fn(key, model, params, batch):
    imgs, _ = batch
    recon_imgs, mu, sigma = model.apply({'params': params}, key, imgs)
    # loss is negative ELBO(L)
    eps = 1e-6
    recon_imgs = jnp.clip(recon_imgs, eps, 1.0 - eps) # Just to be safe
    recon_loss = -jnp.sum(imgs * jnp.log(recon_imgs) + (1 - imgs)  * jnp.log(1 - recon_imgs), axis=(1, 2)) # Binary cross entropy loss
    recon_loss = jnp.mean(recon_loss)
    kl_loss = -0.5 * jnp.sum(1 + jnp.log(jnp.square(sigma)) - jnp.square(mu) - jnp.square(sigma), axis=-1)
    kl_loss = jnp.mean(kl_loss)
    return recon_loss + kl_loss

class TrainerModule:
    def __init__(self, latent_dim, train_dataloader, val_dataloader, seed=42):
        self.latent_dim = latent_dim
        self.seed = seed
        self.init_key = jax.random.key(self.seed)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.example_img = next(iter(self.train_dataloader))[0][:8]
        self.model = VAE(latent_dim)
        self.logger = SummaryWriter()
        
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
        optimizer = optax.adagrad(0.02)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)


    def train_model(self, key, num_epochs=150): # Originally 1500
        best_loss = 1e6
        for epoch_idx in tqdm(range(num_epochs)):
            key, subkey = jax.random.split(key)
            self.train_epoch(subkey, epoch_idx)
            if epoch_idx % 10 == 0:
                key, eval_key = jax.random.split(key)
                eval_loss = self.eval_model(eval_key, self.val_dataloader)
                self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
                if eval_loss < best_loss:
                    best_loss = eval_loss
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
            