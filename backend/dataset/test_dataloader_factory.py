import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.dataset.dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders()

print("=" * 50)
print("RAASTA DataLoader Factory")
print("=" * 50)

print("Train batches :", len(train_loader))
print("Validation batches :", len(val_loader))

images, masks = next(iter(train_loader))

print()

print("Image Shape :", images.shape)
print("Mask Shape  :", masks.shape)