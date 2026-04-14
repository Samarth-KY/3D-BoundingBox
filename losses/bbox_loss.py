import torch
import torch.nn as nn
import torch.nn.functional as F

# Edge pairs grouped by parallel direction, each parallel edge should be equal length
EDGE_GROUPS = [
    [(0,1), (2,3), (4,5), (6,7)], # direction 1
    [(1,2), (0,3), (5,6), (4,7)], # direction 2
    [(0,4), (1,5), (2,6), (3,7)], # vertical edges
]

def compute_edge_consistency_loss(corners: torch.Tensor) -> torch.Tensor:
    """
    Penalizes predicted corners that don't form a valid cuboid.
    Within each group of 4 parallel edges,each parallel edge should be equal length
    
    corners: [B, 8, 3]
    returns: scalar loss
    """
    total = torch.tensor(0.0, device=corners.device)

    for group in EDGE_GROUPS:
        # Compute length of each edge in this group
        lengths = []
        for i, j in group:
            edge_vec = corners[:, i, :] - corners[:, j, :] # [B, 3]
            length = torch.norm(edge_vec, dim=-1) # [B]
            lengths.append(length)
        
        lengths = torch.stack(lengths, dim=1) # [B, 4]
        # Variance across the 4 parallel edges should be 0 for a perfect cuboid
        total = total + lengths.std(dim=1).mean() # Use std instead of var so the scale matches the corner loss

    return total / len(EDGE_GROUPS)

def compute_diagonal_loss(pred:torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    """
    Captures overall box size/scale
    pred, gt: [B, 8, 3]
    """
    # Consider diagonally opposite corners
    pred_diagonal = torch.norm(pred[:, 0] - pred[:, 6], dim=1) 
    gt_diagonal = torch.norm(gt[:, 0] - gt[:, 6], dim=1) 
    return F.smooth_l1_loss(pred_diagonal, gt_diagonal)

class BBoxLoss(nn.Module):
    def __init__(self, lambda_edge: float = 0.1, lambda_diagonal: float = 0.1):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_diagonal = lambda_diagonal

    def forward(self, pred_corners: torch.Tensor, gt_corners: torch.Tensor):
        """
        pred_corners: [B, 8, 3]
        gt_corners: [B, 8, 3]
        returns: total_loss, corner_loss, edge_loss
        """
        corner_loss = F.smooth_l1_loss(pred_corners, gt_corners)
        edge_loss   = compute_edge_consistency_loss(pred_corners)
        diagonal_loss = compute_diagonal_loss(pred_corners, gt_corners)
        total_loss  = corner_loss + self.lambda_edge * edge_loss + self.lambda_diagonal* diagonal_loss

        return total_loss, corner_loss, edge_loss, diagonal_loss
    
