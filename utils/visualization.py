import sys
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import *
from data.preprocessing import list_scenes, load_scene, convert_xyz_to_points

def overlay_mask(rgb: np.ndarray, masks:np.ndarray, alpha:float = 0.5) -> np.ndarray:
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Exepected mask shape [N, H, W] or [H, W], got {masks.shape}")

    n_instances = masks.shape[0]
    cmap = plt.get_cmap("tab20", max(n_instances, 1))

    masked_rgb = rgb.astype(np.float32).copy()
    
    for i in range(n_instances):
        m = masks[i].astype(bool)
        if not m.any():
            continue
        color = np.array(cmap(i)[:3], dtype=np.float32)*255.0
        masked_rgb[m] = (1.0 - alpha) * masked_rgb[m] + alpha * color
    
    return np.clip(masked_rgb, 0, 255).astype(np.uint8)

def build_bbox_linesets(bboxes: np.ndarray, color: tuple = [0.0, 1.0, 0.0]) -> list[o3d.geometry.LineSet]:
    if bboxes.ndim != 3 or bboxes.shape[1:] != (8, 3):
        raise ValueError(f"Expected bbox shape [N, 8, 3], got {bboxes.shape}")
     
    # Cuboid connectivity for 8 corner points.
    box_edges = np.array(
	[
		[0, 1],
		[1, 2],
		[2, 3],
		[3, 0],
		[4, 5],
		[5, 6],
		[6, 7],
		[7, 4],
		[0, 4],
		[1, 5],
		[2, 6],
		[3, 7],
	],
	dtype=np.int32,
    )
    color = np.array(color)

    line_sets = []
    for i, box in enumerate(bboxes):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(box.astype(np.float64))
        ls.lines = o3d.utility.Vector2iVector(box_edges)
        ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(box_edges), 1)))
        line_sets.append(ls)
    return line_sets

def print_scene_stats(scene_dir: Path, xyz, bbox, masks, rgb):
    print(f"\n--- {scene_dir.name}")
    print(f"RGB shape:        {rgb.shape}")
    print(f"Point cloud shape:{xyz.shape}")
    print(f"Masks shape:      {masks.shape}")
    print(f"BBoxes shape:     {bbox.shape}  ({bbox.shape[0]} objects)")
    print(f"PC Z range:       [{xyz[2].min():.3f}, {xyz[2].max():.3f}]")
    print(f"BBox center range: x={bbox[:,:,0].mean():.3f}, y={bbox[:,:,1].mean():.3f}, z={bbox[:,:,2].mean():.3f}")

def visualize_scene(scene_dir: Path, keep_indices: list[int] = None, title: str = None) -> None:
    xyz, bbox, masks, rgb = load_scene(scene_dir)

    if keep_indices is not None:
        selected_masks = masks[keep_indices]
        selected_bbox = bbox[keep_indices]
    else:
        selected_masks = masks
        selected_bbox = bbox

    masked_rgb = overlay_mask(rgb, selected_masks)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{scene_dir.name} - {title}")
    ax[0].imshow(rgb)
    ax[0].set_title("Original RGB")
    ax[0].axis("off")
    ax[1].imshow(masked_rgb)
    ax[1].set_title("Selected Masks")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()

    # Flatten XYZ map into Nx3 points and align RGB colors to the same flattening order.
    points = convert_xyz_to_points(xyz)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    combined_mask = np.any(selected_masks.astype(bool), axis=0).reshape(-1)

    # Drop invalid points (NaN/Inf/zeros).
    scene_valid = np.isfinite(points).all(axis=1) & (points != 0).any(axis=1)
    if keep_indices is not None:
        keep = combined_mask & scene_valid
    else:
        keep = scene_valid

    points = points[keep]
    colors = colors[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    bbox_lines = build_bbox_linesets(selected_bbox)

    # Add a small coordinate frame at camera origin.
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, frame, *bbox_lines])

def _build_scene_pcd(xyz: np.ndarray, rgb: np.ndarray,
                     keep_indices: list = None,
                     masks: np.ndarray = None) -> o3d.geometry.PointCloud:
    """
    Internal helper: flatten XYZ map, drop invalid points, optionally filter
    to only points belonging to selected instance masks.
    """
    points = convert_xyz_to_points(xyz)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
 
    # Drop NaN / Inf / zero-depth points.
    valid = np.isfinite(points).all(axis=1) & (points != 0).any(axis=1)
 
    if keep_indices is not None and masks is not None:
        selected_masks = masks[keep_indices]
        instance_mask = np.any(selected_masks.astype(bool), axis=0).reshape(-1)
        valid = valid & instance_mask
 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[valid])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    return pcd

def visualize_predictions(
    scene_dir: Path,
    instance_indices: list,
    pred_corners_world: np.ndarray,
    gt_corners_world: np.ndarray,
    per_instance_errors_mm: list = None,
) -> None:
    """
    Visualize model predictions vs GT bounding boxes for a scene.
 
    Args:
        scene_dir: Path to the scene directory.
        instance_indices: List of instance IDs shown (used for mask selection).
        pred_corners_world: [N, 8, 3] predicted corners in world coordinates.
        gt_corners_world: [N, 8, 3] GT corners in world coordinates.
        per_instance_errors_mm: Optional list of per-instance errors in mm for console output.
    """
    xyz, bbox, masks, rgb = load_scene(scene_dir)
 
    # 2D: RGB + selected mask overlay
    selected_masks = masks[instance_indices]
    masked_rgb = overlay_mask(rgb, selected_masks)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{scene_dir.name}: Predictions (red) vs GT (green)")
    ax[0].imshow(rgb);        ax[0].set_title("Original RGB");    ax[0].axis("off")
    ax[1].imshow(masked_rgb); ax[1].set_title("Instance Masks"); ax[1].axis("off")
    plt.tight_layout()
    plt.show()
 
    # Console: per-instance error summary
    if per_instance_errors_mm is not None:
        print(f"\nScene: {scene_dir.name}")
        for inst_id, err_mm in zip(instance_indices, per_instance_errors_mm):
            print(f"Instance {inst_id:2d}: mean corner error = {err_mm:.1f} mm")
 
    # 3D: point cloud + GT (green) + predictions (red)
    pcd = _build_scene_pcd(xyz, rgb, instance_indices, masks)
 
    gt_lines   = build_bbox_linesets(gt_corners_world,   color=[0.0, 1.0, 0.0])  # green
    pred_lines = build_bbox_linesets(pred_corners_world, color=[1.0, 0.0, 0.0])  # red
 
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
 
    o3d.visualization.draw_geometries(
        [pcd, frame, *gt_lines, *pred_lines],
        window_name=f"{scene_dir.name} Green=GT  Red=Pred",
    )

def main():
    
    scenes = list_scenes(raw_data_dir) # raw_data_dir defined in config

    for scene in scenes[:10]:
        visualize_scene(scene)

if __name__ == "__main__":
    main()