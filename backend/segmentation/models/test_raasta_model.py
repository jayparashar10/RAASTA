import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from backend.segmentation.models.raasta_model import RAASTAModel

model = RAASTAModel()

dummy = torch.randn(4, 3, 512, 512)

with torch.no_grad():
    output = model(dummy)

print("=" * 50)
print("RAASTA Model Test")
print("=" * 50)
print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)