import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.segmentation.models.segformer import load_segformer
from backend.training.checkpoint import CheckpointManager
from backend.configs.config import CHECKPOINT_PATH

model = load_segformer()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

manager = CheckpointManager(CHECKPOINT_PATH)

manager.save(
    model,
    optimizer,
    epoch=1,
    loss=0.75,
) 