from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepglobe_dataset import DeepGlobeDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "DeepGlobe" / "train"

dataset = DeepGlobeDataset(DATASET_PATH)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

print("=" * 50)
print("RAASTA DataLoader Test")
print("=" * 50)

images, masks = next(iter(dataloader))

print("Batch Images Shape :", images.shape)
print("Batch Masks Shape  :", masks.shape)