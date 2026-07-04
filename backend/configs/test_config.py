import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.configs.config import *

print("=" * 50)
print("RAASTA Configuration")
print("=" * 50)

print("Dataset :", DATASET_PATH)
print("Train :", TRAIN_PATH)
print("Validation :", VALID_PATH)
print("Checkpoint :", CHECKPOINT_PATH)

print()

print("Batch Size :", BATCH_SIZE)
print("Epochs :", EPOCHS)
print("Learning Rate :", LEARNING_RATE)

print()

print("Device :", DEVICE)