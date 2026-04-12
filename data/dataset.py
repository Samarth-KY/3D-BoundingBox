import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Ensure project-root imports work even when running this file directly from data/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import (
    load_sample,
    convert_xyz_to_points,
    extract_instance_points,
    sample_points,
    compute_centroid,
    center_points,
    center_corners,
    list_samples,
)
from config.config import raw_data_dir

def get_split(sample_dirs: list, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """
    Returns (train_dirs, val_dirs, test_dirs) , split at scene level.
    """
    if len(sample_dirs) == 0:
        raise ValueError("No sample directories provided")
    
    np.random.seed(seed)
    dirs = sample_dirs.copy()
    np.random.shuffle(dirs)

    num_samples = len(dirs)
    val_size = int(num_samples*val_ratio)
    test_size = int(num_samples*test_ratio)

    test_dirs = dirs[:test_size]
    remaining = dirs[test_size:]

    val_dirs = remaining[:val_size]
    train_dirs = remaining[val_size:]

    return train_dirs, val_dirs, test_dirs
    

class BBoxDataset(Dataset):
    def __init__(self, sample_dirs: list[Path], n_points: int = 1024, augment: bool = False):
        """
        Flattens scenes into per-instance samples at init time.
        """
        super().__init__()

        self.instances = []
        self.n_points = n_points
        self.augment = augment

        for sample_dir in sample_dirs:
            xyz, bbox, masks, rgb = load_sample(sample_dir)

            num_instances = masks.shape[0]
            for i in range(num_instances):
                self.instances.append((sample_dir, i))

    def __len__(self):
        # list of (sample_dir, instance_idx)
        return len(self.instances)

    def __getitem__(self, idx):
        """
        Returns dict with keys: 'points', 'corners', 'centroid' as torch float32 tensors.
        """
        sample_dir, instance_idx = self.instances[idx]

        xyz, bbox, masks, rgb = load_sample(sample_dir)

        points          = convert_xyz_to_points(xyz)
        instance_points = extract_instance_points(points, masks, instance_idx)
        sampled_points  = sample_points(instance_points, self.n_points)
        centroid        = compute_centroid(sampled_points)
        centered_points = center_points(sampled_points, centroid)

        corners          = bbox[instance_idx]
        centered_corners = center_corners(corners, centroid)

        points = torch.from_numpy(centered_points).float()
        corners = torch.from_numpy(centered_corners).float()
        centroid = torch.from_numpy(centroid).float()

        return {
            "points": points,
            "corners": corners,
            "centroid": centroid
        }


if __name__ == "__main__":
    sample_dirs = list_samples(raw_data_dir)
    train_dirs, val_dirs, test_dirs = get_split(sample_dirs=sample_dirs)

    dataset = BBoxDataset(train_dirs)
    print(f"Total instances: {len(dataset)}")

    sample = dataset[0]
    print(f"points shape:  {sample['points'].shape}")   # expect (1024, 3)
    print(f"corners shape: {sample['corners'].shape}")  # expect (8, 3)
    print(f"centroid:      {sample['centroid']}")        # expect (3,)

    # Sanity check: centered points should have near-zero mean
    print(f"points mean (should be ~0): {sample['points'].mean(axis=0)}")

    # Sanity check: corners should also be near the origin
    print(f"corners mean: {sample['corners'].mean(axis=0)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"batch points: {batch['points'].shape}")   # expect (4, 1024, 3)
    print(f"batch corners: {batch['corners'].shape}") # expect (4, 8, 3)