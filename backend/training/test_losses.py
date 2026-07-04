import torch

from losses import CombinedLoss

criterion = CombinedLoss()

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

loss = criterion(
    predictions,
    targets,
)

print("=" * 50)
print("RAASTA Loss Test")
print("=" * 50)

print(loss)