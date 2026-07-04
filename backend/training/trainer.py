import torch

from backend.training.metrics import dice_score, iou_score


class Trainer:
    """
    RAASTA Training Engine
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0
        total_dice = 0
        total_iou = 0

        for batch_idx, (images, masks) in enumerate(self.train_loader):

            if batch_idx % 10 == 0:
                print(f"Training Batch {batch_idx}/{len(self.train_loader)}")

            images = images.to(self.device)
            masks = masks.float().to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            logits = self.model(images)

            # Calculate Loss
            loss = self.criterion(
                logits,
                masks,
            )

            # Backpropagation
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            total_dice += dice_score(
                logits.detach(),
                masks,
            )

            total_iou += iou_score(
                logits.detach(),
                masks,
            )

        n = len(self.train_loader)

        return (
            total_loss / n,
            total_dice / n,
            total_iou / n,
        )

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0
        total_dice = 0
        total_iou = 0

        for batch_idx, (images, masks) in enumerate(self.val_loader):

            if batch_idx % 10 == 0:
                print(f"Validation Batch {batch_idx}/{len(self.val_loader)}")

            images = images.to(self.device)
            masks = masks.float().to(self.device)

            # Forward Pass
            logits = self.model(images)

            # Calculate Loss
            loss = self.criterion(
                logits,
                masks,
            )

            total_loss += loss.item()

            total_dice += dice_score(
                logits,
                masks,
            )

            total_iou += iou_score(
                logits,
                masks,
            )

        n = len(self.val_loader)

        return (
            total_loss / n,
            total_dice / n,
            total_iou / n,
        )