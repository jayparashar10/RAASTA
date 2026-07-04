from pathlib import Path


class SatelliteDatasetLoader:
    """
    Generic Dataset Loader

    This class will later support:
    - DeepGlobe
    - SpaceNet
    - Massachusetts Roads
    - OpenStreetMap
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def dataset_exists(self):
        return self.dataset_path.exists()

    def get_dataset_path(self):
        return self.dataset_path

    def print_info(self):
        print("=" * 50)
        print("RAASTA Dataset Loader")
        print("=" * 50)
        print(f"Dataset Path : {self.dataset_path}")
        print(f"Exists       : {self.dataset_exists()}")