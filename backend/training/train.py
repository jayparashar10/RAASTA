import sys
from pathlib import Path

import torch

# ---------------------------------
# Project Root
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# ---------------------------------
# Imports
# ---------------------------------

from backend.configs.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_PATH,
)

from backend.dataset.dataloader import create_dataloaders
from backend.segmentation.models.raasta_model import RAASTAModel
from backend.training.losses import CombinedLoss
from backend.training.trainer import Trainer
from backend.training.logger import TrainingLogger
from backend.training.checkpoint import CheckpointManager


# ---------------------------------
# Main
# ---------------------------------

def main():

    print("=" * 60)
    print("RAASTA AI Training Engine")
    print("=" * 60)

    print("\nLoading Dataset...")

    train_loader, val_loader = create_dataloaders()

    print("Loading RAASTA Model...")

    model = RAASTAModel()

    print("Creating Optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = CombinedLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
    )

    checkpoint = CheckpointManager(
        CHECKPOINT_PATH
    )

    print("\nEverything Loaded Successfully!")

    for epoch in range(EPOCHS):

        TrainingLogger.epoch_start(
            epoch + 1,
            EPOCHS,
        )

        train_loss, train_dice, train_iou = trainer.train_one_epoch()

        TrainingLogger.train(
            train_loss,
            train_dice,
            train_iou,
        )

        val_loss, val_dice, val_iou = trainer.validate()

        TrainingLogger.validation(
            val_loss,
            val_dice,
            val_iou,
        )

        checkpoint.save(
            model,
            optimizer,
            epoch + 1,
            val_loss,
        )

        TrainingLogger.checkpoint_saved()

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()