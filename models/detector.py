import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pointnet2_utils import PointNetSetAbstraction

class BBoxDetector(nn.Module):
    def __init__(self):
        super(BBoxDetector, self).__init__()

        # SA1: anchor_xyz provides instance pts (npoints not used when anchors are provided)
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False
        )

        # SA2: 1024 -> 256
        # l1_points has 128 features, so in_channel = 128 + 3
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=48, in_channel=128+3, mlp=[128, 128, 256], group_all=False
        )

        # SA3: 256 -> 64
        # l2_points has 256 features, so in_channel = 256 + 3
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=64, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False
        )

        # SA4: global pooling, 64 -> 1
        # l3_points has 512 features, so in_channel = 512 + 3
        # group_all=True, so npoint, radius and nsample are ignored
        self.sa4 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 512, 1024], group_all=True
        )

        # Regression head: 1024 -> 24 (8 corners × 3 coords)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 24) # 8 corners × 3 = 24

    def forward(self, scene_points, instance_points):
        """
        Input:
            scene_points: [B, N, 3]
            instance_points: [B, M, 3]
        Output:
            corners: [B, 8, 3]
        """
        B = scene_points.shape[0]

        # DataLoader gives [B, N, 3], SA layers expect [B, 3, N]
        xyz = scene_points.permute(0, 2, 1) # [B, 3, N]

        # SA1: [B, 3, N] -> l1_xyz: [B, 3, 1024], l1_points: [B, 128, 1024]
        l1_xyz, l1_points = self.sa1(xyz, None, anchor_xyz=instance_points)

        # SA2: -> l2_xyz: [B, 3, 256], l2_points: [B, 256, 256]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        # SA3: -> l3_xyz: [B, 3, 64], l3_points: [B, 512, 64]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # SA4: -> l4_xyz: [B, 3, 1], l4_points: [B, 1024, 1]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)


        # Flatten global feature: [B, 1024, 1] -> [B, 1024]
        x = l4_points.view(B, 1024)

        # MLP regression head
        x = self.drop1(F.relu(self.bn1(self.fc1(x)))) # [B, 512]
        x = self.drop2(F.relu(self.bn2(self.fc2(x)))) # [B, 256]
        x = self.fc3(x) # [B, 24]

        # Reshape to corner format
        corners = x.view(B, 8, 3) # [B, 8, 3]
        return corners

    def set_onnx_export(self, flag: bool):
        """Switch all SA layers to ONNX-safe uniform sampling."""
        for module in self.modules():
            if isinstance(module, PointNetSetAbstraction):
                module.onnx_export = flag


if __name__ == "__main__":
    import torch
    model = BBoxDetector()
    dummy_scene_points = torch.randn(4, 8192, 3)
    dummy_instance_points = torch.randn(4, 1024, 3)
    out = model(dummy_scene_points, dummy_instance_points)
    print(f"Output shape: {out.shape}") 

    from losses.bbox_loss import BBoxLoss

    loss_fn = BBoxLoss(lambda_edge=0.1)

    pred = torch.randn(4, 8, 3)
    gt = torch.randn(4, 8, 3)

    total, corner_l, edge_l = loss_fn(pred, gt)
    print(f"Total loss: {total.item():.4f}")
    print(f"Corner loss: {corner_l.item():.4f}")
    print(f"Edge loss:{edge_l.item():.4f}")

    # Sanity check
    total2, corner_l2, _ = loss_fn(gt, gt)
    print(f"Corner loss when pred==gt:{corner_l2.item():.6f}")