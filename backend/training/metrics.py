import torch


def dice_score(predictions, targets, threshold=0.5):
    """
    Calculate Dice Score.
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    targets = targets.float()

    intersection = (predictions * targets).sum()

    dice = (
        2 * intersection
    ) / (
        predictions.sum() + targets.sum() + 1e-8
    )

    return dice.item()


def iou_score(predictions, targets, threshold=0.5):
    """
    Calculate IoU.
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    targets = targets.float()

    intersection = (predictions * targets).sum()

    union = (
        predictions.sum()
        + targets.sum()
        - intersection
    )

    iou = intersection / (union + 1e-8)

    return iou.item()