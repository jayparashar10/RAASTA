from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class DeepGlobeDataset(Dataset):
    """
    PyTorch Dataset for DeepGlobe Road Extraction
    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform

        # Get all satellite images
        self.image_paths = sorted(self.dataset_path.glob("*_sat.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        mask_path = self.dataset_path / image_path.name.replace(
            "_sat.jpg",
            "_mask.png"
        )

        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Apply augmentations (later)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask