class TrainingLogger:
    """
    Simple console logger for RAASTA.
    """

    @staticmethod
    def epoch_start(epoch, total_epochs):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{total_epochs}")
        print("=" * 60)

    @staticmethod
    def train(loss, dice, iou):
        print(
            f"Train | Loss: {loss:.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}"
        )

    @staticmethod
    def validation(loss, dice, iou):
        print(
            f"Valid | Loss: {loss:.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}"
        )

    @staticmethod
    def checkpoint_saved():
        print("✓ Checkpoint Saved")