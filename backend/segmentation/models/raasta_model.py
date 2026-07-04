import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.segmentation.models.segformer import load_segformer


class RAASTAModel(nn.Module):
    """
    Wrapper around SegFormer for binary road segmentation.
    """

    def __init__(self):
        super().__init__()

        self.model = load_segformer()

    def forward(self, images):

        outputs = self.model(pixel_values=images)

        logits = outputs.logits

        # Resize logits to match input image size
        logits = F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Return only Road channel
        logits = logits[:, 1, :, :]

        return logits