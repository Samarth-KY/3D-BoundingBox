import torch
from torch.utils.data import DataLoader
from pathlib import Path
import time
import sys
import wandb
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data.preprocessing import list_scenes
from data.dataset import BBoxDataset, get_split
from models.detector import BBoxDetector
from losses.bbox_loss import BBoxLoss
from config.config import raw_data_dir, valid_instances_json_dir, model_checkpoint_dir

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Run one full pass over the training set.
    Returns dict of average losses for this epoch.
    """
    model.train()
    total_loss_sum = 0.0
    corner_loss_sum = 0.0
    edge_loss_sum = 0.0
    diagonal_loss_sum = 0.0

    for batch in loader:
        scene_points    = batch["scene_points"].to(device) 
        instance_points = batch["instance_points"].to(device) 
        corners = batch["corners"].to(device)  

        optimizer.zero_grad()
        pred_corners = model(scene_points, instance_points)        

        total_loss, corner_loss, edge_loss, diagonal_loss = loss_fn(pred_corners, corners)
        total_loss.backward()

        # Gradient clipping (prevents exploding gradients early in training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss_sum += total_loss.item()
        corner_loss_sum += corner_loss.item()
        edge_loss_sum += edge_loss.item()
        diagonal_loss_sum += diagonal_loss.item()

    n = len(loader)
    return {
        "total": total_loss_sum / n,
        "corner": corner_loss_sum / n,
        "edge": edge_loss_sum / n,
        "diagonal": diagonal_loss_sum / n
    }

@torch.no_grad()
def val_one_epoch(model, loader, loss_fn, device):
    """
    Run one full pass over the validation set, no gradients.
    Returns dict of average losses.
    """
    model.eval()
    total_loss_sum = 0.0
    corner_loss_sum = 0.0
    edge_loss_sum = 0.0
    diagonal_loss_sum = 0.0

    for batch in loader:
        scene_points = batch["scene_points"].to(device)
        instance_points = batch["instance_points"].to(device)
        corners = batch["corners"].to(device)

        pred_corners = model(scene_points, instance_points)

        total_loss, corner_loss, edge_loss, diagonal_loss = loss_fn(pred_corners, corners)
        total_loss_sum += total_loss.item()
        corner_loss_sum += corner_loss.item()
        edge_loss_sum += edge_loss.item()
        diagonal_loss_sum += diagonal_loss.item()

    n = len(loader)
    return {
        "total": total_loss_sum / n,
        "corner": corner_loss_sum / n,
        "edge": edge_loss_sum / n,
        "diagonal": diagonal_loss_sum / n
    }

def main():
    parser = argparse.ArgumentParser(description="Train 3D BBox Detector")
    parser.add_argument("--n_scene_points", type=int, default=8192)
    parser.add_argument("--n_instance_points", type=int, default=1024)
    parser.add_argument("--scene_context", action="store_true", help="Use scene points as context for PointNet2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_edge", type=float, default=0.1)
    parser.add_argument("--lambda_diagonal", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", help="Enable training data augmentation")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    args = parser.parse_args()

    model_checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Wandb config
    config = vars(args)
    config["weight_decay"] = 1e-4
    config["scheduler_patience"] = 10
    config["scheduler_factor"] = 0.5
    wandb.init(project="3d-bbox-detector", config=config, name=args.run_name)

    # Load data and generate dataset
    scene_dirs = list_scenes(raw_data_dir)
    train_dirs, val_dirs, _ = get_split(scene_dirs)

    train_dataset = BBoxDataset(train_dirs, 
                                n_scene_points=args.n_scene_points, 
                                n_instance_points=args.n_instance_points,
                                valid_instances_path=valid_instances_json_dir,
                                scene_context=args.scene_context,
                                augment=args.augment)
    val_dataset = BBoxDataset(val_dirs, 
                              n_scene_points=args.n_scene_points, 
                              n_instance_points=args.n_instance_points, 
                              valid_instances_path=valid_instances_json_dir,
                              scene_context=args.scene_context,
                              augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train instances: {len(train_dataset)}, Val instances: {len(val_dataset)}")

    # Load model, loss, optimizer
    model   = BBoxDetector().to(device)
    loss_fn = BBoxLoss(lambda_edge=args.lambda_edge, lambda_diagonal=args.lambda_diagonal)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_losses = val_one_epoch(model, val_loader, loss_fn, device)

        scheduler.step(val_losses["total"])

        elapsed = time.time() - t0

        # Log to console
        print(
            f"Epoch {epoch:03d}/{args.num_epochs} | "
            f"Train: {train_losses['total']:.4f}"
            f"(corner={train_losses['corner']:.4f}, edge={train_losses['edge']:.4f}, diagonal={train_losses['diagonal']:.4f}) | "
            f"Val: {val_losses['total']:.4f}"
            f"(corner={val_losses['corner']:.4f}, edge={val_losses['edge']:.4f}, diagonal={val_losses['diagonal']:.4f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/total": train_losses["total"],
            "train/corner": train_losses["corner"],
            "train/edge": train_losses["edge"],
            "train/diagonal": train_losses["diagonal"],
            "val/total": val_losses["total"],
            "val/corner": val_losses["corner"],
            "val/edge": val_losses["edge"],
            "val/diagonal": val_losses["diagonal"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": elapsed,
        })

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, model_checkpoint_dir / f"best_model_{args.run_name}.pth")
            print(f"Saved best model (val_loss={best_val_loss:.4f})")

    wandb.finish()
    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()