from torch.utils.data import DataLoader, Subset
import torch

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


def create_dataloaders():
    """
    Creates train and validation dataloaders
    using a reproducible 90/10 split.
    """

    train_dataset = DeepGlobeDataset(
        TRAIN_PATH,
        transform=get_train_transforms(),
    )

    val_dataset = DeepGlobeDataset(
        TRAIN_PATH,
        transform=get_validation_transforms(),
    )

    dataset_size = len(train_dataset)

    indices = torch.randperm(
        dataset_size,
        generator=torch.Generator().manual_seed(42)
    )

    split = int(0.9 * dataset_size)

    train_indices = indices[:split]
    val_indices = indices[split:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader