import jax
import torch


def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    return torch.from_numpy(imgs.copy()).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
