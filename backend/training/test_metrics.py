import torch

from metrics import dice_score, iou_score

predictions = torch.randn(
    2,
    512,
    512,
)

targets = torch.randint(
    0,
    2,
    (
        2,
        512,
        512,
    ),
)

print("=" * 50)
print("RAASTA Metrics Test")
print("=" * 50)

print("Dice :", dice_score(predictions, targets))
print("IoU  :", iou_score(predictions, targets))