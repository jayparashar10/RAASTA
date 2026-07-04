from pathlib import Path

from deepglobe_dataset import DeepGlobeDataset
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.preprocessing.transforms import get_train_transforms
from deepglobe_dataset import DeepGlobeDataset

DATASET_PATH = PROJECT_ROOT / "datasets" / "DeepGlobe" / "train"

dataset = DeepGlobeDataset(
    DATASET_PATH,
    transform=get_train_transforms()
)

print("=" * 50)
print("RAASTA Dataset Test")
print("=" * 50)

print("Total Images :", len(dataset))

image, mask = dataset[0]

print("Image Shape :", image.shape)
print("Mask Shape  :", mask.shape)

print("Image Type  :", image.dtype)
print("Mask Type   :", mask.dtype)