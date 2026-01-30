from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import optax
from flax.training import train_state, checkpoints
import os
from tqdm import tqdm

#Dataset
class XORDataset(Dataset):
    def __init__(self, size, seed, std=0.1):
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.std = std
        self.generate_data()
            
    def generate_data(self):
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        label = (data.sum(axis=1) == 1).astype(np.int32)
        
        data += self.np_rng.normal(loc=0., scale=self.std, size=data.shape)
        
        self.data = data
        self.label = label
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return self.data[i], self.label[i]


def plot_samples(data, label):
    zeros = data[(label==0)]
    ones = data[(label==1)]
    
    fig = plt.figure(figsize=(4, 4), dpi=100)
    plt.scatter(zeros[:, 0], zeros[:, 1], edgecolors="#333", label="Class 0")
    plt.scatter(ones[:, 0], ones[:, 1], edgecolors="#333", label="Class 1")
    plt.title("XOR dataset")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
    return fig

# plot_samples(dataset.data, dataset.label)
# plt.show()

# Dataloader
# For jax, since DataLoader defaults to Pytorch tensors
def numpy_collate(x):
    if isinstance(x[0], np.ndarray): # for images
        return np.stack(x)
    elif isinstance(x[0], (tuple, list)): # for initial call
        x = zip(*x)
        return [numpy_collate(item) for item in x]
    else: # for labels
        return np.array(x)
    

# Model
class SimpleClassifier(nn.Module):
    num_hiddens : int
    num_outputs : int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_hiddens)(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.num_outputs)(x)
        return x

# Loss
def calculate_loss_acc(state, params, batch):
    data, labels = batch
    logits = state.apply_fn(params, data).squeeze(axis=-1)
    pred = (logits>0).astype(jnp.float32)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred == labels).mean()
    return loss, acc

# Training
@jax.jit
def train_step(state, batch):
    (loss, acc), grads = jax.value_and_grad(calculate_loss_acc, argnums=1, has_aux=True)(state, state.params, batch) # Has additional output
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

@jax.jit
def eval_step(state, batch):
    loss, acc = calculate_loss_acc(state, state.params, batch)
    return acc

def train_model(state, dataloader, num_epochs=100):
    for epoch in tqdm(range(num_epochs)):
        accs = []
        for batch in dataloader:
            state, loss, acc = train_step(state, batch)
            accs.append(acc)
        # print(epoch, np.mean(accs))
    return state

def eval_model(state, dataloader):
    accs, batch_sizes = [], [] # account for batch size differences
    for batch in dataloader:
        state, loss, acc = train_step(state, batch)
        accs.append(acc)
        batch_sizes.append(batch[0].shape[0])
    
    acc = sum([a*b for a, b in zip(accs, batch_sizes)]) / sum(batch_sizes)
    print(f"Evaluation accuracy: {100.0*acc:.2f}")
    

dataset = XORDataset(size=2500, seed=0)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)
eval_dataset = XORDataset(size=500, seed=42)
eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False, drop_last=False, collate_fn=numpy_collate)
    
model = SimpleClassifier(8, 1)
rng = jax.random.key(0)
key1, key2 = jax.random.split(rng, 2)
inp = jax.random.uniform(key1, (8, 2))
params = model.init(key2, inp)

# Optimizer
optimizer = optax.sgd(learning_rate=0.1)

# Bundle
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
state = train_model(state, dataloader)

checkpoints.save_checkpoint(ckpt_dir=os.path.abspath('checkpoints/'), target=state, step=100, prefix="XORmodel", overwrite=True)

# loaded_state = checkpoints.restore_checkpoint(ckpt_dir='checkpoints/', target=state, prefix='XORmodel')

eval_model(state, eval_dataloader) # Already 100% at 15th epoch

trained_model = model.bind(state.params)

def visualize_classification(model, data, label):
    fig = plot_samples(data, label)
    
    c0 = np.array(to_rgba("C0"))
    c1 = np.array(to_rgba("C1"))
    x1 = np.arange(-0.5, 1.5, step=0.01)
    x2 = np.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = np.meshgrid(x1, x2, indexing='ij')
    model_inputs = np.stack([xx1, xx2], axis=-1)
    logits = model(model_inputs)
    pred = nn.sigmoid(logits)
    output_image = (1 - pred) * c0[None, None] + pred * c1[None, None]
    output_image = jax.device_get(output_image)
    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig
    

_ = visualize_classification(trained_model, dataset.data, dataset.label)
plt.show()