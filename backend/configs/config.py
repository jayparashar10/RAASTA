from pathlib import Path
import torch
import os

# -------------------------
# Project Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Detect Google Colab
if os.path.exists("/content/drive/MyDrive/RAASTA/DeepGlobe"):
    DATASET_PATH = Path("/content/drive/MyDrive/RAASTA/DeepGlobe")
    print("Running on Google Colab")
else:
    DATASET_PATH = PROJECT_ROOT / "datasets" / "DeepGlobe"
    print("Running on Local Machine")

TRAIN_PATH = DATASET_PATH / "train"
VALID_PATH = DATASET_PATH / "valid"      # Not used anymore, but kept for compatibility
TEST_PATH = DATASET_PATH / "test"

CHECKPOINT_PATH = PROJECT_ROOT / "backend" / "checkpoints"
LOG_PATH = PROJECT_ROOT / "backend" / "logs"

# -------------------------
# Training
# -------------------------

IMAGE_SIZE = 512

BATCH_SIZE = 4

NUM_WORKERS = 2

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

# -------------------------
# Device
# -------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# -------------------------
# Classes
# -------------------------

NUM_CLASSES = 2