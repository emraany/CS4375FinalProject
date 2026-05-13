"""Dataset loaders for MNIST and CIFAR-10.

Downloads land in ./datasets/ to avoid colliding with this package's ./data/
module name. Training set is split 90/10 into train/val deterministically.
The val subset uses test-style transforms (no augmentation).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


DATA_ROOT = "./datasets"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


DATASET_INFO = {
    "mnist": {
        "in_channels": 1,
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)],
    },
    "cifar10": {
        "in_channels": 3,
        "num_classes": 10,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "frog", "dog", "horse", "ship", "truck",
        ],
    },
}


def _mnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    # Pad 28x28 to 32x32 so LeNet-5 spatial sizes work directly.
    base = [
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ]
    train = transforms.Compose(base)
    test = transforms.Compose(base)
    return train, test


def _cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train, test


def _build_datasets(name: str):
    name = name.lower()
    if name == "mnist":
        train_tf, test_tf = _mnist_transforms()
        train_aug = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=train_tf)
        train_plain = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=test_tf)
        test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=test_tf)
    elif name == "cifar10":
        train_tf, test_tf = _cifar10_transforms()
        train_aug = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
        train_plain = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=test_tf)
        test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Expected 'mnist' or 'cifar10'.")
    return train_aug, train_plain, test


def get_dataloaders(
    dataset: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train, val, and test DataLoaders plus dataset metadata."""
    name = dataset.lower()
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    train_aug, train_plain, test = _build_datasets(name)

    n_total = len(train_aug)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    split = random_split(range(n_total), [n_train, n_val], generator=gen)
    train_indices = list(split[0])
    val_indices = list(split[1])

    train_set = Subset(train_aug, train_indices)
    val_set = Subset(train_plain, val_indices)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, DATASET_INFO[name]
