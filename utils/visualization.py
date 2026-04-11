import os
import sys
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def list_samples(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def load_sample(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.load(sample_dir / "pc.npy")
    bbox = np.load(sample_dir / "bbox3d.npy")
    masks = np.load(sample_dir / "mask.npy")
    rgb = np.asarray(Image.open(sample_dir / "rgb.jpg").convert("RGB"))

    return xyz, bbox, masks, rgb

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

def convert_xyz_to_points(xyz: np.ndarray) -> np.ndarray:
    # Dataset format: [3, H, W], where channels are x,y,z
    if xyz.ndim == 3 and xyz.shape[0] == 3:
        xyz = np.moveaxis(xyz, 0, -1) # [3, H, W] -> [H, W, 3]
        points = xyz.reshape(-1, 3) # [H, W, 3] -> [H*W, 3]
    elif xyz.ndim == 3 and xyz.shape[-1] == 3:
        points = xyz.reshape(-1, 3) # [H, W, 3] -> [H*W, 3]
    else:
        raise ValueError(f"Unsupported point cloud shape: {xyz.shape}")
    return points

def build_bbox_linesets(bboxes: np.ndarray, color: tuple = [0.0, 1.0, 0.0]) -> list[o3d.geometry.LineSet]:
    if bboxes.ndim != 3 or bboxes.shape[1:] != (8, 3):
        raise ValueError(f"Expected bbox shape [N, 8, 3], got {bboxes.shape}")
     
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

def print_sample_stats(sample_dir: Path, xyz, bbox, masks, rgb):
    print(f"\n--- {sample_dir.name} ---")
    print(f"RGB shape:        {rgb.shape}")
    print(f"Point cloud shape:{xyz.shape}")
    print(f"Masks shape:      {masks.shape}")
    print(f"BBoxes shape:     {bbox.shape}  ({bbox.shape[0]} objects)")
    print(f"PC Z range:       [{xyz[2].min():.3f}, {xyz[2].max():.3f}]")
    print(f"BBox center range: x={bbox[:,:,0].mean():.3f}, y={bbox[:,:,1].mean():.3f}, z={bbox[:,:,2].mean():.3f}")

def visualize_sample(sample_dir):

    xyz, bbox, masks, rgb = load_sample(sample_dir)

    masked_rgb = overlay_mask(rgb, masks)

    fig, ax= plt.subplots(1, 2)
    ax[0].imshow(rgb)
    ax[0].set_title("Original RGB Image")

    ax[1].imshow(masked_rgb)
    ax[1].set_title("Masked RGB Image")
    
    plt.show()

    points = convert_xyz_to_points(xyz)
    colors = rgb.reshape(-1, 3).astype(np.float32)/255.0

    valid = np.isfinite(points).all(axis=1) & (points != 0).any(axis=1)
    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    bbox_lines = build_bbox_linesets(bbox)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd,frame, *bbox_lines])

    print_sample_stats(sample_dir, xyz, bbox, masks, rgb)

def main():
    
    raw_data_dir = PROJECT_ROOT / "data" / "raw_data"
    samples = list_samples(raw_data_dir)

    for sample in samples[:10]:
        visualize_sample(sample)

if __name__ == "__main__":
    main()