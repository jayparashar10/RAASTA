import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from backend.segmentation.models.segformer import load_segformer

print("=" * 50)
print("Loading SegFormer...")
print("=" * 50)

model = load_segformer()

print(model.__class__.__name__)

print("\nModel loaded successfully!")