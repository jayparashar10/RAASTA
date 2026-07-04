from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Dataset path
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "DeepGlobe" / "train"

print("Project Root:", PROJECT_ROOT)
print("Dataset Path:", DATASET_PATH)
print("Dataset Exists:", DATASET_PATH.exists())

# Find first satellite image
satellite_images = sorted(DATASET_PATH.glob("*_sat.jpg"))

print(f"Found {len(satellite_images)} satellite images")

# Pick first image
image_path = satellite_images[0]

# Find corresponding mask
mask_path = DATASET_PATH / image_path.name.replace("_sat.jpg", "_mask.png")

print(f"Image : {image_path.name}")
print(f"Mask  : {mask_path.name}")

# Load image
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load mask
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

# Display
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Satellite Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Road Mask")
plt.axis("off")

plt.tight_layout()
plt.show()