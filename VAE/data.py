import numpy as np
import torch
import torchvision


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32) / 255.0
    return (img >= 0.5).astype(np.float32) # Binarize


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch, axis=0)
    elif isinstance(batch[0], (tuple, list)):
        return [numpy_collate(x) for x in zip(*batch)]
    else:
        return np.array(batch)


def load_data(seed, batch_size, num_workers):
    
    train_dataset = torchvision.datasets.MNIST("../data", train=True, transform=image_to_numpy, download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths=[50000, 10000], generator=torch.Generator().manual_seed(seed))
    test_dataset = torchvision.datasets.MNIST("../data", train=False, transform=image_to_numpy, download=True)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=numpy_collate, pin_memory=True, persistent_workers=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=numpy_collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=numpy_collate)

    return (train_dataset, val_dataset, test_dataset), (train_dataloader, val_dataloader, test_dataloader)

