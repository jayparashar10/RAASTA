import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.dataset.dataloader import create_dataloaders
from backend.segmentation.models.segformer import load_segformer
from backend.configs.config import DEVICE

print("=" * 60)
print("RAASTA Sanity Check")
print("=" * 60)

# Load one batch
train_loader, _ = create_dataloaders()
images, masks = next(iter(train_loader))

print(f"Images Shape : {images.shape}")
print(f"Masks Shape  : {masks.shape}")

# Load model
model = load_segformer().to(DEVICE)
model.eval()

images = images.to(DEVICE)

with torch.no_grad():
    outputs = model(pixel_values=images)

print(f"Raw Logits Shape : {outputs.logits.shape}")

print("\nSanity Check Passed!")