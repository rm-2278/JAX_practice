import numpy as np
import jax
import torchvision

from utils import jax_to_torch


class GenerateCallback:
    def __init__(self, input_imgs, every_n_epochs=1):
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs
        
    def log_generation(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            reconst_img = model.apply({'params': state.params}, self.input_imgs)
            # Move to device
            reconst_img = jax.device_get(reconst_img)
            
            # Save imgs
            imgs = np.stack([self.input_imgs, reconst_img], axis=1).reshape(-1, *reconst_img.shape[1:])
            imgs = jax_to_torch(imgs)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(-1, 1))
            # add to logger
            logger.add_image("Reconstructions", grid, global_step=epoch)
