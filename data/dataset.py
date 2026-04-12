import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json

# Ensure project-root imports work even when running this file directly from data/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import (
    load_scene,
    convert_xyz_to_points,
    extract_instance_points,
    sample_points,
    compute_centroid,
    center_points,
    center_corners,
    list_scenes,
    normalize_points
)
from config.config import raw_data_dir, valid_instances_json_dir

def get_split(scene_dirs: list, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """
    Returns (train_dirs, val_dirs, test_dirs) , split at scene level.
    """
    if len(scene_dirs) == 0:
        raise ValueError("No scene directories provided")
    
    np.random.seed(seed)
    dirs = scene_dirs.copy()
    np.random.shuffle(dirs)

    num_scenes = len(dirs)
    val_size = int(num_scenes*val_ratio)
    test_size = int(num_scenes*test_ratio)
    
    # Order: test | val | train
    test_dirs = dirs[:test_size]
    remaining = dirs[test_size:]

    val_dirs = remaining[:val_size]
    train_dirs = remaining[val_size:]

    return train_dirs, val_dirs, test_dirs
    

class BBoxDataset(Dataset):
    def __init__(self, scene_dirs: list[Path], n_points: int = 1024, valid_instances_path:str = None, augment: bool = False):
        """
        Flattens scenes into per-instance scenes at init time.
        """
        super().__init__()

        self.instances = []
        self.n_points = n_points
        self.augment = augment

        if valid_instances_path:
            scene_dir_paths = {str(d) for d in scene_dirs}
            with open(valid_instances_path) as f:
                valid_instances = json.load(f)
            self.instances = [(Path(s), i) for s, i in valid_instances if s in scene_dir_paths]
        else:
            for scene_dir in scene_dirs:
                masks = np.load(scene_dir / "mask.npy")
                for i in range(masks.shape[0]):
                    self.instances.append((scene_dir, i))

    def __len__(self):
        # list of (scene_dir, instance_idx)
        return len(self.instances)

    def __getitem__(self, idx):
        """
        Returns dict with keys: 'points', 'corners', 'centroid' as torch float32 tensors.
        """
        scene_dir, instance_idx = self.instances[idx]

        xyz, bbox, masks, rgb = load_scene(scene_dir)

        points          = convert_xyz_to_points(xyz)
        instance_points = extract_instance_points(points, masks, instance_idx)
        sampled_points  = sample_points(instance_points, self.n_points)
        centroid        = compute_centroid(sampled_points)
        centered_points = center_points(sampled_points, centroid)
        normalized_points, scale   = normalize_points(centered_points)

        corners          = bbox[instance_idx]
        centered_corners = center_corners(corners, centroid)
        centered_corners /= scale

        points = torch.from_numpy(normalized_points).float()
        corners = torch.from_numpy(centered_corners).float()
        centroid = torch.from_numpy(centroid).float()

        return {
            "points": points,
            "corners": corners,
            "centroid": centroid,
            "scale": torch.tensor(scale, dtype=torch.float32)
        }


if __name__ == "__main__":
    scene_dirs = list_scenes(raw_data_dir)
    train_dirs, val_dirs, test_dirs = get_split(scene_dirs=scene_dirs)

    dataset = BBoxDataset(train_dirs, n_points=2048, valid_instances_path=valid_instances_json_dir)
    print(f"Total instances: {len(dataset)}")

    scene = dataset[0]
    print(f"points shape:  {scene['points'].shape}")   
    print(f"corners shape: {scene['corners'].shape}")  
    print(f"centroid:      {scene['centroid']}")      

    # Sanity check: centered points should have near-zero mean
    print(f"points mean (should be ~0): {scene['points'].mean(axis=0)}")
    print(f"points max : {scene['points'].max()}")

    # Sanity check: corners should also be near the origin
    print(f"corners mean: {scene['corners'].mean(axis=0)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"batch points: {batch['points'].shape}")  
    print(f"batch corners: {batch['corners'].shape}")