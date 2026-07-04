import torch
from pathlib import Path


class CheckpointManager:
    """
    Handles saving and loading model checkpoints.
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        epoch,
        loss,
        filename="latest_checkpoint.pth",
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        save_path = self.checkpoint_dir / filename

        torch.save(checkpoint, save_path)

        print(f"Checkpoint saved -> {save_path}")

    def load(
        self,
        model,
        optimizer,
        filename="latest_checkpoint.pth",
    ):

        load_path = self.checkpoint_dir / filename

        checkpoint = torch.load(load_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Checkpoint loaded <- {load_path}")

        return checkpoint["epoch"], checkpoint["loss"]