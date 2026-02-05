from dataclasses import dataclass, field
from typing import List


@dataclass #generates boilerplate
class TrainConfig:
    dataset_path: str = "../data"
    checkpoint_path: str = "../checkpoints/autoencoder"
    batch_size: int = 256
    num_workers: int = 4
    c_hid: int = 32
    latent_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 384]) # Creates a new list per instance
    lr: float = 1e-3
    seed: int = 42
    prefetch_size: int = 2
