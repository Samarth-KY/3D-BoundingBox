import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import (
    load_scene,
    convert_xyz_to_points,
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
    def __init__(self, scene_dirs: list[Path], 
                 n_scene_points: int, 
                 n_instance_points:int = 1024, 
                 valid_instances_path:str = None, 
                 scene_context:bool = True, 
                 augment: bool = False):
        super().__init__()

        self.instances = []
        self.n_instance_points = n_instance_points
        self.n_scene_points = n_scene_points
        assert n_scene_points > n_instance_points, f"n_scene_points ({n_scene_points}) must be greater than n_instance_points ({n_instance_points})"

        self.scene_context = scene_context
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
        Returns dict with keys: 'scene_points', 'instance_points', 'corners', 'centroid' as torch float32 tensors.
        """
        scene_dir, instance_idx = self.instances[idx]
        xyz, bbox, masks, rgb = load_scene(scene_dir)

        # Full scene points: [H*W, 3]
        all_points = convert_xyz_to_points(xyz)
        # Separate instance points from scene points
        instance_mask = masks[instance_idx].flatten().astype(bool)  # [H*W]
        instance_points = all_points[instance_mask]
        background_points = all_points[~instance_mask]

        # Sample each group independently
        sampled_instance_points, _ = sample_points(instance_points, self.n_instance_points)

        if self.scene_context:
            n_background_points = self.n_scene_points - self.n_instance_points
            sampled_background_points, _ = sample_points(background_points, n_background_points)
            
            # Merge: sampled instance points + sampled background points
            # IMP: instance points must be a subset of sampled scene points. (will be used as anchor points for Pointnet2 predictions)
            sampled_scene_points = np.concatenate([sampled_instance_points, sampled_background_points], axis=0)
            # Shuffle so the network doesn't learn positional bias from the ordering
            shuffle_idx = np.random.permutation(self.n_scene_points)
            sampled_scene_points = sampled_scene_points[shuffle_idx]
        else:
            sampled_scene_points = sampled_instance_points

        # Center and normalize using instance points
        centroid = compute_centroid(sampled_instance_points)

        centered_instance = center_points(sampled_instance_points, centroid)
        normalized_instance, scale = normalize_points(centered_instance)

        centered_scene = center_points(sampled_scene_points, centroid)
        if scale != 0:
            normalized_scene = centered_scene / scale
        else:
            normalized_scene = centered_scene

        corners = bbox[instance_idx]
        centered_corners = center_corners(corners, centroid)
        if scale != 0:
            normalized_corners = centered_corners / scale
        else:
            normalized_corners = centered_corners
        
        if self.augment:
            # Random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos, sin = np.cos(theta), np.sin(theta)
            R = np.array([
                [cos,-sin, 0],
                [sin, cos, 0],
                [0, 0, 1],
            ], dtype=np.float32)

            normalized_scene = normalized_scene @ R.T
            normalized_instance = normalized_instance @ R.T
            normalized_corners = normalized_corners @ R.T

            # Random uniform scaling
            scale_factor = np.random.uniform(0.9, 1.1)
            normalized_scene = normalized_scene * scale_factor
            normalized_instance = normalized_instance *scale_factor
            normalized_corners = normalized_corners *scale_factor

            # Random jitter
            normalized_scene = normalized_scene + np.clip(np.random.normal(0, 0.005, normalized_scene.shape), -0.02, 0.02).astype(np.float32)
            normalized_instance = normalized_instance + np.clip(np.random.normal(0, 0.005, normalized_instance.shape), -0.02, 0.02).astype(np.float32)

        # Convert to tensors
        scene_points_t = torch.from_numpy(normalized_scene).float()
        instance_points_t = torch.from_numpy(normalized_instance).float()
        corners_t = torch.from_numpy(normalized_corners).float()
        centroid_t = torch.from_numpy(centroid).float()

        return {
            "scene_points": scene_points_t,
            "instance_points": instance_points_t,
            "corners": corners_t,
            "centroid": centroid_t,
            "scale": torch.tensor(scale, dtype=torch.float32),
        }

if __name__ == "__main__":
    scene_dirs = list_scenes(raw_data_dir)
    train_dirs, val_dirs, test_dirs = get_split(scene_dirs=scene_dirs)

    dataset = BBoxDataset(train_dirs, n_scene_points=4096, n_instance_points=1024, valid_instances_path=valid_instances_json_dir)
    print(f"Total instances: {len(dataset)}")

    scene = dataset[0]
    print(f"scene_points shape: {scene['scene_points'].shape}")
    print(f"instance_points shape: {scene['instance_points'].shape}")
    print(f"corners shape: {scene['corners'].shape}")
    print(f"centroid: {scene['centroid']}")
    print(f"scale: {scene['scale']}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"batch scene_points: {batch['scene_points'].shape}")
    print(f"batch instance_points: {batch['instance_points'].shape}")
    print(f"batch corners: {batch['corners'].shape}")