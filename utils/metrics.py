import torch

def mean_corner_distance(pred_corners, gt_corners, scales, centroids):
    """
    Computes mean L2 corner error in world coordinates (meters).
    
    pred_corners: [B, 8, 3] normalized centroid-relative
    gt_corners: [B, 8, 3] normalized centroid-relative
    scales: [B] scale factors from normalize_points
    centroids: [B, 3] centroids in world coords
    
    Returns: scalar, mean error in meters
    """
    scales = scales.view(-1, 1, 1)
    centroids = centroids.unsqueeze(1)

    pred_world = pred_corners * scales + centroids # [B, 8, 3]
    gt_world = gt_corners * scales + centroids # [B, 8, 3]
    distances = torch.norm(pred_world - gt_world, dim=-1) # [B, 8]
    return distances.mean().item()

def per_instance_corner_distance(pred_corners, gt_corners, scales, centroids):
    """
    Same as mean_corner_distance but returns per-instance errors.
    
    Returns: np.ndarray of shape [B], errors in meters
    """
    scales = scales.view(-1, 1, 1)
    centroids = centroids.unsqueeze(1)

    pred_world = pred_corners * scales + centroids
    gt_world = gt_corners * scales + centroids
    distances = torch.norm(pred_world - gt_world, dim=-1) # [B, 8]
    return distances.mean(dim=-1).cpu().numpy() # [B]

def recall_at_threshold(errors_m, threshold_m):
    """
    Fraction of instances with mean corner error below threshold.
    
    errors_m: np.ndarray of per-instance errors in meters
    threshold_m: float, threshold in meters
    
    Returns: float in [0, 1]
    """
    return float((errors_m < threshold_m).mean())