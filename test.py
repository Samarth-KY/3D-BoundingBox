import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
import sys, os
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import (
    list_scenes, load_scene, convert_xyz_to_points,
    extract_instance_points, sample_points,
    compute_centroid, center_points, normalize_points,
)
from data.dataset import BBoxDataset, get_split
from models.detector import BBoxDetector
from utils.metrics import per_instance_corner_distance, recall_at_threshold
from utils.visualization import visualize_predictions
from config.config import raw_data_dir, valid_instances_json_dir, model_checkpoint_dir, results_dir

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run model inference over entire loader, collect per-instance errors in world coords.
    Returns array of shape [N_instances] with errors in meters.
    """
    model.eval()
    all_errors = []

    for batch in loader:
        scene_points = batch["scene_points"].to(device)
        instance_points = batch["instance_points"].to(device)
        corners = batch["corners"].to(device)
        scales = batch["scale"].to(device)
        centroids = batch["centroid"].to(device)

        pred_corners = model(scene_points, instance_points)

        errors = per_instance_corner_distance(pred_corners, corners, scales, centroids)
        all_errors.append(errors)

    return np.concatenate(all_errors) # [N_total]

@torch.no_grad()
def run_inference_on_scene(model, device, scene_dir, valid_instance_ids, n_scene_points, n_instance_points):
    """
    Run the model on all valid instances of one scene.
    Returns:
        pred_corners_world: np.ndarray [N, 8, 3]
        gt_corners_world: np.ndarray [N, 8, 3]
        errors_mm: list[float]
    """
    model.eval()
    xyz, bbox, masks, rgb = load_scene(scene_dir)
    all_points = convert_xyz_to_points(xyz)
 
    pred_corners_world = []
    gt_corners_world = []
    errors_mm = []
 
    for inst_id in valid_instance_ids:
        gt_corners = bbox[inst_id] # [8, 3] world coords
 
        # Preprocessing
        inst_mask = masks[inst_id].flatten().astype(bool)
        inst_pts = all_points[inst_mask]
        bg_pts = all_points[~inst_mask]
 
        if len(inst_pts) < 10:
            print(f"  Instance {inst_id}: too few points, skipping")
            continue
 
        sampled_inst, _ = sample_points(inst_pts, n_instance_points)
 
        n_bg = n_scene_points - n_instance_points
        sampled_bg, _ = sample_points(bg_pts, n_bg)
        scene_pts = np.concatenate([sampled_inst, sampled_bg], axis=0)
        shuffle_idx = np.random.permutation(n_scene_points)
        scene_pts = scene_pts[shuffle_idx]
 
        centroid = compute_centroid(sampled_inst)
        centered_inst = center_points(sampled_inst, centroid)
        normed_inst, scale = normalize_points(centered_inst)
        centered_scene = center_points(scene_pts, centroid)
        normed_scene = centered_scene / scale if scale != 0 else centered_scene
 
        # Inference
        scene_t = torch.from_numpy(normed_scene).float().unsqueeze(0).to(device)
        inst_t  = torch.from_numpy(normed_inst).float().unsqueeze(0).to(device)
 
        pred_norm = model(scene_t, inst_t).squeeze(0).cpu().numpy() # [8, 3]
        pred_world = pred_norm * scale + centroid # [8, 3]
 
        err_mm = np.linalg.norm(pred_world - gt_corners, axis=-1).mean() * 1000
 
        pred_corners_world.append(pred_world)
        gt_corners_world.append(gt_corners)
        errors_mm.append(err_mm)
 
    return (
        np.stack(pred_corners_world), # [N, 8, 3]
        np.stack(gt_corners_world), # [N, 8, 3]
        errors_mm,
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D bbox models on test set")
    parser.add_argument("--n_scene_points", type=int, default=8192)
    parser.add_argument("--n_instance_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--visualize", action="store_true", help="After evaluation, run inference+visualization on N test scenes")
    parser.add_argument("--n_vis_scenes", type=int, default=5, help="Number of test scenes to visualize (only used with --visualize)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(exist_ok=True)

    # Test split (use same seed as training)
    scene_dirs = list_scenes(raw_data_dir)
    _, _, test_dirs = get_split(scene_dirs)

    test_dataset = BBoxDataset(
        test_dirs,
        n_scene_points=args.n_scene_points,
        n_instance_points=args.n_instance_points,
        valid_instances_path=valid_instances_json_dir,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    print(f"Test instances: {len(test_dataset)}")


    with open(valid_instances_json_dir) as f:
        valid_instances = json.load(f)
    valid_by_scene = {}
    for scene_str, inst_id in valid_instances:
        valid_by_scene.setdefault(scene_str, []).append(inst_id)

    for model_name in os.listdir(model_checkpoint_dir):
        # Load model
        model = BBoxDetector().to(device)
        ckpt  = torch.load(model_checkpoint_dir/ model_name, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

        # Run evaluation
        errors = evaluate(model, test_loader, device)  # [N] in meters

        # Summary metrics
        print(f"Test Results for model: {model_name}")
        print(f"Instances evaluated:  {len(errors)}")
        print(f"Mean corner error:    {errors.mean()*1000:.1f} mm")
        print(f"Median corner error:  {np.median(errors)*1000:.1f} mm")
        print(f"Std corner error:     {errors.std()*1000:.1f} mm")
        print(f"Best  (5th pct):      {np.percentile(errors, 5)*1000:.1f} mm")
        print(f"Worst (95th pct):     {np.percentile(errors, 95)*1000:.1f} mm")
        print(f"Recall @ 20mm:        {recall_at_threshold(errors, 0.020):.1%}")
        print(f"Recall @ 50mm:        {recall_at_threshold(errors, 0.050):.1%}")
        print(f"Recall @ 100mm:       {recall_at_threshold(errors, 0.100):.1%}")

        # Save results
        results = {
            "checkpoint_epoch": ckpt["epoch"],
            "val_loss_at_checkpoint": ckpt["val_loss"],
            "n_test_instances": len(errors),
            "mean_error_mm": float(errors.mean()* 1000),
            "median_error_mm": float(np.median(errors) * 1000),
            "std_error_mm": float(errors.std() * 1000),
            "p5_error_mm": float(np.percentile(errors, 5) * 1000),
            "p95_error_mm": float(np.percentile(errors, 95) * 1000),
            "recall_20mm": recall_at_threshold(errors, 0.020),
            "recall_50mm": recall_at_threshold(errors, 0.050),
            "recall_100mm": recall_at_threshold(errors, 0.100),
        }

        save_dir = results_dir / model_name.replace(".pth", ".json")
        with open(save_dir, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_dir}")

        if args.visualize:
            print(f"Visualizing {args.n_vis_scenes} test scenes for {model_name} ...")
            shown = 0
            for scene_dir in test_dirs:
                scene_key = str(scene_dir)
                if scene_key not in valid_by_scene:
                    continue
 
                inst_ids = valid_by_scene[scene_key]
 
                pred_world, gt_world, errs_mm = run_inference_on_scene(
                    model, device, scene_dir, inst_ids,
                    args.n_scene_points, args.n_instance_points,
                )
 
                visualize_predictions(
                    scene_dir=scene_dir,
                    instance_indices=inst_ids[:len(pred_world)],
                    pred_corners_world=pred_world,
                    gt_corners_world=gt_world,
                    per_instance_errors_mm=errs_mm,
                )
 
                shown += 1
                if shown >= args.n_vis_scenes:
                    break

if __name__ == "__main__":
    main()