from torch.utils.data import DataLoader, random_split

from backend.dataset.deepglobe_dataset import DeepGlobeDataset
from backend.preprocessing.transforms import (
    get_train_transforms,
    get_validation_transforms,
)

from backend.configs.config import (
    TRAIN_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
)

import torch


def create_dataloaders():
    """
    Creates train and validation dataloaders
    using a 90/10 split of the training dataset.
    """

    full_dataset = DeepGlobeDataset(
        TRAIN_PATH,
        transform=get_train_transforms(),
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Apply validation transforms to validation subset
    val_dataset.dataset.transform = get_validation_transforms()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader