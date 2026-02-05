import os
import numpy as np
from tqdm.auto import tqdm

import jax

import optax
from flax import jax_utils
from flax.training import train_state, checkpoints

from torch.utils.tensorboard import SummaryWriter

from model import AutoEncoder
from callbacks import GenerateCallback

# Loss
def mse_loss(model, params, batch): 
    imgs, _ = batch
    recon_imgs = model.apply({'params': params}, imgs)
    loss = ((imgs - recon_imgs)**2).mean(axis=0).sum() #mean over batch, sum over pixel
    return loss

# Training functionality
class TrainerModule:
    def __init__(self, c_hid, latent_dim, checkpoint_path, train_dataloader, val_dataloader, lr=1e-3, seed=42, prefetch_size=2):
        self.c_hid = c_hid
        self.latent_dim = latent_dim
        self.lr = lr
        self.seed = seed
        self.prefetch_size = prefetch_size
        self.checkpoint_path = checkpoint_path
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.model = AutoEncoder(c_hid, latent_dim)
        
        self.log_dir = os.path.abspath( # orbax tensorstore refuse relative path
            os.path.join(self.checkpoint_path, f'cifar10_{latent_dim}')
        )
        
        self.example_imgs = next(iter(self.val_dataloader))[0][:8] # For initialising generator, model
        self.generator_callback = GenerateCallback(self.example_imgs, every_n_epochs=50)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        
        self.create_functions()
        
        self.init_model()

    def _prefetch(self, dataloader):
        if self.prefetch_size and self.prefetch_size > 0:
            if jax.device_count() > 1:
                return jax_utils.prefetch_to_device(dataloader, size=self.prefetch_size)
            return (jax.device_put(batch) for batch in dataloader)
        return dataloader
        
    def create_functions(self): # Create jitted functions
        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(self.model, params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params) # Or argnums
            state = state.apply_gradients(grads=grads)
            return state, loss
        self.train_step = jax.jit(train_step)
        
        def eval_step(state, batch):
            return mse_loss(self.model, state.params, batch)
        self.eval_step = jax.jit(eval_step)

    def init_model(self):
        rngs = jax.random.key(self.seed)
        rngs, init_rngs = jax.random.split(rngs)
        params = self.model.init(init_rngs, self.example_imgs)['params']
        lr_schedular = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=100,
            decay_steps=500*len(self.train_dataloader),
            end_value=1e-5
        )
        optimizer = optax.chain(optax.clip(1), optax.adam(lr_schedular))
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
    
    def train_model(self, num_epochs=500):
        best_loss = 1e6
        for epoch_idx in tqdm(range(num_epochs)):
            self.train_epoch(epoch = epoch_idx)
            if epoch_idx % 10 == 0:
                eval_loss = self.eval_model(self.val_dataloader)
                self.logger.add_scalar('val/loss', eval_loss, global_step=epoch_idx)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_model(step=epoch_idx)
                self.generator_callback.log_generation(self.model, self.state, self.logger, epoch_idx)
                self.logger.flush()
        
    def train_epoch(self, epoch):
        losses = []
        for batch in self._prefetch(self.train_dataloader):
            self.state, loss = self.train_step(self.state, batch)
            losses.append(loss)
        losses = np.stack(jax.device_get(losses))
        avg_loss = losses.mean()
        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
        

    def eval_model(self, dataloader):
        losses = []
        batch_sizes = []
        for batch in self._prefetch(dataloader):
            loss = self.eval_step(self.state, batch)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss
    
    def save_model(self, step=0): # Save model during training
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, step=step, target=self.state.params, prefix=f'cifar10_{self.latent_dim}_')
    
    def load_model(self, pretrained=False): # When loading trained or pretrained model
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f'cifar10_{self.latent_dim}_')
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.checkpoint_path, f'cifar10_{self.latent_dim}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)
    
    def checkpoint_exists(self):
        return checkpoints.latest_checkpoint(self.log_dir, prefix=f'cifar10_{self.latent_dim}_') is not None
        # return os.path.isfile(os.path.join(self.checkpoint_path, f'cifar10_{self.latent_dim}.ckpt'))
