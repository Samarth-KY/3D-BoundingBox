import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points: indexed points data, [B, S, C] or [B, S, K, C]
    """
    B, N, C = points.shape

    if idx.dim() == 2:
        # idx: [B, S]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)   # [B, S, C]
        new_points = torch.gather(points, 1, idx_expanded)   # [B, S, C]
        return new_points

    elif idx.dim() == 3:
        # idx: [B, S, K]
        S = idx.shape[1]
        K = idx.shape[2]

        idx_flat = idx.reshape(B, S * K)                              # [B, S*K]
        idx_flat_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, C) # [B, S*K, C]

        gathered = torch.gather(points, 1, idx_flat_expanded)        # [B, S*K, C]
        new_points = gathered.reshape(B, S, K, C)                    # [B, S, K, C]
        return new_points

    else:
        raise ValueError(f"Unsupported idx shape: {idx.shape}")

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def uniform_point_sample(xyz, npoint):
    """
    ONNX-friendly deterministic sampler.
    Selects npoint indices uniformly along the point index axis.

    Input:
        xyz: [B, N, 3]
        npoint: int
    Return:
        idx: [B, npoint] (long)
    """
    N = xyz.shape[1]
    device = xyz.device

    # Build npoint indices from [0, N-1]
    idx1d = torch.linspace(0, N - 1, steps=npoint, device=device)
    idx1d = idx1d.round().to(torch.long)
    idx1d = torch.clamp(idx1d, 0, N - 1) # [npoint]

    # Add batch axis and broadcast
    idx = idx1d.unsqueeze(0) # [1, npoint]
    idx = idx.expand(xyz.shape[0], npoint) # [B, npoint]

    return idx

def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between each two points.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist = dist + torch.sum(src ** 2, dim=-1, keepdim=True)
    dist = dist + torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(1, 2)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    radius2 = radius * radius

    inf = torch.full_like(sqrdists, 1e10)
    masked_dists = torch.where(sqrdists <= radius2, sqrdists, inf)  # [B, S, N]

    selected_dists, group_idx = torch.topk(
        masked_dists,
        k=nsample,
        dim=-1,
        largest=False,
        sorted=False
    )  # [B, S, nsample]

    nearest_idx = torch.argmin(sqrdists, dim=-1, keepdim=True)  # [B, S, 1]

    invalid_mask = selected_dists >= 1e10
    group_idx = torch.where(invalid_mask, nearest_idx, group_idx)

    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, anchor_xyz=None, returnfps=False, onnx_export: bool = False):
    """
    Input:
        npoint:     number of centroids (ignored if anchor_xyz is provided)
        radius:     ball query radius
        nsample:    max neighbours per ball
        xyz:        all points positions, [B, N, 3]
                    Neighbours are searched in this cloud.
        points:     all points features, [B, N, D] or None
                    Features are gathered from this using neighbour indices.
        anchor_xyz: pre-computed centroid positions, [B, S, 3] or None
                    If provided, used as centroids instead of running FPS
                    on xyz.  Neighbours are still gathered from xyz/points.
    Return:
        new_xyz:    centroid positions, [B, S, 3]
        new_points: grouped & centroid-relative data, [B, S, nsample, 3+D]
    """
    B, N, C = xyz.shape

    if anchor_xyz is not None:
        new_xyz = anchor_xyz # [B, S, 3]
    else:
        fps_idx = uniform_point_sample(xyz, npoint) if onnx_export else farthest_point_sample(xyz, npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)

    S = new_xyz.shape[1]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.reshape(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.onnx_export = False  # set True during ONNX export to avoid FPS

    def forward(self, xyz, points, anchor_xyz=None):
        """
        Input:
            xyz:        input points position data, [B, C, N]
                        The cloud in which neighbours are searched.
            points:     input points data, [B, D, N] or None
                        Features to gather for each neighbour.
            anchor_xyz: pre-computed centroids, [B, S, C] or None
                        If provided, these are used as centroids instead
                        of running FPS on xyz.  Ignored when group_all=True.
        Return:
            new_xyz:            sampled points position data, [B, C, S]
            new_points_concat:  sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # Permute from channel-first [B, C, N] to [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample,
                xyz, points,
                anchor_xyz=anchor_xyz,
                onnx_export=getattr(self, 'onnx_export', False)
            )
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.amax(new_points, dim=2)
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        self.onnx_export = False  # set True during ONNX export to avoid FPS

    def forward(self, xyz, points, anchor_xyz=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if anchor_xyz is None:
            S = self.npoint
            fps_idx = uniform_point_sample(xyz, S) if getattr(self, 'onnx_export', False) else farthest_point_sample(xyz, S)
            new_xyz = index_points(xyz, fps_idx)
        else:
            # Use pre-computed centroids (e.g. instance anchor points)
            new_xyz = anchor_xyz  # [B, S, C]

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=3)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            new_points = torch.amax(grouped_points, dim=2, keepdim=True)  # [B, D_i, 1, S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)

        if len(new_points_list) == 1:
            new_points_concat = new_points_list[0]
        else:
            new_points_concat = torch.cat(new_points_list, dim=1)  # [B, sum(D_i), 1, S]

        new_points_concat = new_points_concat.squeeze(2)  # [B, sum(D_i), S]

        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)   # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)   # [B, S, C]
        points2 = points2.permute(0, 2, 1)  # [B, S, D]

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # [B, N, S]

            # Distances should never be negative, clamp for numerical stability
            dists = torch.clamp(dists, min=0.0)

            # Use only 3 nearest neighbors
            dists, idx = torch.sort(dists, dim=-1)
            dists = dists[:, :, :3]   # [B, N, 3]
            idx = idx[:, :, :3]       # [B, N, 3]

            # Clamp again before reciprocal to avoid division by zero / tiny values
            dists = torch.clamp(dists, min=1e-6)

            dist_recip = 1.0 / dists
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            norm = torch.clamp(norm, min=1e-6)
            weight = dist_recip / norm  # [B, N, 3]

            neighbor_points = index_points(points2, idx)  # [B, N, 3, D]
            interpolated_points = torch.sum(
                neighbor_points * weight.unsqueeze(-1),
                dim=2
            )  # [B, N, D]

            # Extra guardrail for ONNX/ORT numerical issues
            interpolated_points = torch.nan_to_num(
                interpolated_points, nan=0.0, posinf=0.0, neginf=0.0
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # [B, D', N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Final guardrail
        new_points = torch.nan_to_num(new_points, nan=0.0, posinf=0.0, neginf=0.0)
        return new_points