import json
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import list_scenes, load_scene, convert_xyz_to_points, extract_instance_points
from config.config import *
from utils.visualization import visualize_scene

# Data cleanup config
BBOX_TOL = 0.05 # enlarges each bbox side by a percentage.
MIN_INSIDE_RATIO = 0.80 # keeps an instance if most points are inside the expanded box.
MIN_VALID_POINTS = 20 # min number of points inside bbox for a valid instance

def build_obb(corners:np.ndarray, tolerance:float) -> o3d.geometry.OrientedBoundingBox:
    if corners.shape != (8, 3):
        raise ValueError(f"Expected corners shape (8, 3), got {corners.shape}")

    corner_cloud = o3d.utility.Vector3dVector(corners.astype(np.float64))
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(corner_cloud)

    # Expand bbox in all dimensions to allow small sensor noise.
    expanded_extent = obb.extent * (1.0 + tolerance)
    expanded_extent = np.maximum(expanded_extent, 1e-6)
    return o3d.geometry.OrientedBoundingBox(obb.center, obb.R, expanded_extent)

def is_valid_instance(
    instance_points: np.ndarray,
    corners: np.ndarray,
    min_inside_ratio: float = MIN_INSIDE_RATIO,
    tolerance: float = BBOX_TOL,
) -> bool:
    """
    Instance is valid iff all finite/non-zero instance points lie inside its oriented bbox.
    """
    if instance_points.ndim != 2 or instance_points.shape[1] != 3:
        raise ValueError(f"Expected instance points shape (N, 3), got {instance_points.shape}")
    if instance_points.shape[0] == 0:
        return False

    valid_points = instance_points[np.isfinite(instance_points).all(axis=1)]
    valid_points = valid_points[(valid_points != 0).any(axis=1)]
    if valid_points.shape[0] < MIN_VALID_POINTS:
        return False

    obb = build_obb(corners, tolerance)
    point_cloud = o3d.utility.Vector3dVector(valid_points.astype(np.float64))
    inside_idx = obb.get_point_indices_within_bounding_box(point_cloud)

    inside_ratio = len(inside_idx) / float(valid_points.shape[0])
    return inside_ratio >= min_inside_ratio

def main():
    scene_dirs = list_scenes(raw_data_dir)
    valid = []
    rejected = []
    valid_by_scene: dict[str, list[int]] = {}
    all_by_scene: dict[str, list[int]] = {}

    for scene_dir in scene_dirs:
        xyz, bbox, masks, rgb = load_scene(scene_dir)
        points = convert_xyz_to_points(xyz)

        scene_key = str(scene_dir)
        all_by_scene[scene_key] = list(range(masks.shape[0]))
        valid_by_scene[scene_key] = []

        for i in range(masks.shape[0]):
            instance_points = extract_instance_points(points, masks, i)
            
            if is_valid_instance(instance_points, bbox[i]):
                valid.append((scene_key, i))
                valid_by_scene[scene_key].append(i)
            else:
                rejected.append((scene_key, i))

    print(f"Valid instances:    {len(valid)}")
    print(f"Rejected instances: {len(rejected)}")

    with open(valid_instances_json_dir, "w") as f:
        json.dump(valid, f, indent=2)
    
    print(f"Saved to {valid_instances_json_dir}")

    # Visualize a few examples before and after cleanup.
    num_vis_scenes = 3
    shown = 0

    for scene_dir in scene_dirs:
        scene_key = str(scene_dir)
        if len(valid_by_scene[scene_key]) == 0 or len(valid_by_scene[scene_key]) == len(all_by_scene[scene_key]):
            continue
        print(f"Scene: {scene_dir.name}")
        visualize_scene(scene_dir, all_by_scene[scene_key], "Before cleanup (all instances)")
        visualize_scene(scene_dir, valid_by_scene[scene_key], "After cleanup (valid instances only)")

        shown += 1
        if shown >= num_vis_scenes:
            break

    if shown == 0:
        print("No mixed scenes found for before/after visualization (all kept or all rejected in each scene).")

if __name__ == "__main__":
    main()