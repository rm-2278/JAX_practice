from typing import Callable, Sequence
import flax.linen as nn

class BaseNN(nn.Module):
    act_fn: Callable
    hidden_dim: Sequence = (512, 256, 256, 128)
    num_classes: int = 10
    kernel_init: Callable = nn.linear.default_kernel_init
    
    @nn.compact
    def __call__(self, return_activation = False):
        x = x.reshape(x.shape[0], -1)        
        activations = []
        for hid in self.hidden_dim:
            x = nn.Dense(hid, kernel_init=self.kernel_init)(x)
            activations.append(x)
            x = self.act_fn(x)
            activations.append(x)
        x = nn.Dense(self.num_classes, kernel_init=self.kernel_init)(x)
        activations.append(x)
        return (x, activations) if return_activation else x