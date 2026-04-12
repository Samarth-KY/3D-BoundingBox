import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d

def list_samples(root: Path) -> list[Path]:
    # Each sample is stored in its own directory; sorting keeps iteration deterministic.
    return sorted([p for p in root.iterdir() if p.is_dir()])

def load_sample(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Expected per-sample files:
    # - pc.npy: dense XYZ map 
    # - bbox3d.npy: 3D bounding box annotations
    # - mask.npy: instance masks [N, H, W]
    # - rgb.jpg: color image aligned with the XYZ map
    xyz = np.load(sample_dir / "pc.npy")
    bbox = np.load(sample_dir / "bbox3d.npy")
    masks = np.load(sample_dir / "mask.npy")
    rgb = np.asarray(Image.open(sample_dir / "rgb.jpg").convert("RGB"))

    return xyz, bbox, masks, rgb

def convert_xyz_to_points(xyz: np.ndarray) -> np.ndarray:
    # Dataset format: [3, H, W], where channels are x,y,z
    if xyz.ndim == 3 and xyz.shape[0] == 3:
        # Channel-first -> channel-last before flattening into a point list.
        xyz = np.moveaxis(xyz, 0, -1)           # [3, H, W] -> [H, W, 3]
        points = xyz.reshape(-1, 3)             # [H, W, 3] -> [H*W, 3]
    elif xyz.ndim == 3 and xyz.shape[-1] == 3:
        # Already channel-last, so only flatten spatial dimensions.
        points = xyz.reshape(-1, 3)             # [H, W, 3] -> [H*W, 3]
    else:
        raise ValueError(f"Unsupported point cloud shape: {xyz.shape}")
    return points

def extract_instance_points(points: np.ndarray, masks:np.ndarray, instance_id: int) -> np.ndarray:
    """
    Given the full scene point cloud and masks, return only the points belonging to the specific instance.
    """
    if masks.ndim != 3:
        raise ValueError(f"Exepected mask shape [N, H, W], got {masks.shape}")
    
    N, H, W = masks.shape
    # points is expected to be flattened from [H, W, 3] to [H*W, 3].
    if points.shape[0] != H * W:
        raise ValueError(f"Mismatch: points has {points.shape[0]} elements, but masks has {H*W} pixels")
    
    if instance_id >= masks.shape[0]:
        raise IndexError("instance_id out of range")

    # Select one instance mask, flatten to match flattened points, then boolean-index.
    instance_mask = masks[instance_id]
    instance_mask = instance_mask.flatten().astype(bool)
    
    instance_points = points[instance_mask]

    return instance_points

def sample_points(points: np.ndarray, n:int = 1024) -> np.ndarray:
    """
    Randomly sample or pad points to match exactly n points
    """
    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("No points available to sample from")
    
    if num_points >= n:
        # Enough points: sample without replacement to avoid duplicates.
        indices = np.random.choice(num_points, size=n, replace=False)
    else:
        # Too few points: sample with replacement to pad up to n points.
        indices = np.random.choice(num_points, size=n, replace=True)

    sampled_points = points[indices]

    return sampled_points

def compute_centroid(points: np.ndarray) -> np.ndarray:
    """
    Returns the mean xyz of the point cloud. Shape: (3,)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")
    
    if points.shape[0] == 0:
        raise ValueError("Cannot compute centroid of empty point set")
    
    return points.mean(axis=0)

def center_points(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Subtract centroid from all points. Returns centered points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")
    
    if centroid.shape != (3,):
        raise ValueError(f"Expected centroid shape (3,), got {centroid.shape}")
    
    return points - centroid

def center_corners(corners: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Subtract centroid from all 8 bbox corners.
    corners shape: (8, 3)
    """
    if corners.ndim != 2 or corners.shape[1] != 3:
        raise ValueError(f"Expected corners shape (8, 3), got {corners.shape}")
    
    if corners.shape[0] != 8:
        raise ValueError(f"Expected 8 corners, got {corners.shape[0]}")
    
    if centroid.shape != (3,):
        raise ValueError(f"Expected centroid shape (3,), got {centroid.shape}")
    
    return corners - centroid


