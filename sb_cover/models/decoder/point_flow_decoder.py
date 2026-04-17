"""PointFlowDecoder: Multi-scale point flow decoder for dense reconstruction.

Propagates center-level predictions to full point cloud resolution using
multi-scale k-NN feature aggregation with residual connections and LayerNorm.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class ResidualScaleMLP(nn.Module):
    """Scale-specific MLP with residual connection.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden dimension.
        out_dim: Output dimension.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.shortcut(x)


class PointFlowDecoder(nn.Module):
    """Multi-scale point flow decoder.

    Uses multiple k-NN scales to aggregate center-level features and
    predict per-point displacements for dense reconstruction.

    Args:
        hidden_dim: Hidden feature dimension (default 384).
        k_neighbors: List of k values for multi-scale aggregation.
        num_centers: Number of input centers.
        center_feat_dim: Dimension of per-center features.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        k_neighbors: List[int] = None,
        num_centers: int = 128,
        center_feat_dim: int = 3,
    ):
        super().__init__()
        if k_neighbors is None:
            k_neighbors = [4, 8, 16]
        self.k_neighbors = k_neighbors
        self.num_centers = num_centers

        # Per-scale feature extractors
        # Input per scale: relative_pos(3) + center_disp(3) + distance(1) = 7
        scale_in_dim = 7
        self.scale_mlps = nn.ModuleList()
        for _ in k_neighbors:
            self.scale_mlps.append(
                ResidualScaleMLP(scale_in_dim, hidden_dim, hidden_dim)
            )

        # LayerNorm before output for stability
        self.pre_output_norm = nn.LayerNorm(hidden_dim * len(k_neighbors))

        # Fusion MLP: combines multi-scale features
        fusion_in = hidden_dim * len(k_neighbors)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def _knn(self, query: Tensor, reference: Tensor, k: int) -> Tensor:
        """K-nearest neighbor search.

        Args:
            query: (B, N, 3).
            reference: (B, M, 3).
            k: Number of neighbors.

        Returns:
            Neighbor indices (B, N, k).
        """
        dist = torch.cdist(query, reference)
        _, idx = dist.topk(k, dim=-1, largest=False)
        return idx

    def _gather_neighbors(
        self,
        noisy_points: Tensor,
        centers: Tensor,
        center_pred: Tensor,
        k: int,
    ):
        """Gather neighbor features at a given k-NN scale.

        Args:
            noisy_points: (B, N, 3).
            centers: (B, M, 3).
            center_pred: (B, M, 3).
            k: Number of neighbors.

        Returns:
            features: (B, N, k, 7).
            idw_base: (B, N, 3) IDW-interpolated base prediction.
        """
        B, N, _ = noisy_points.shape
        M = centers.shape[1]
        k = min(k, M)

        nn_idx = self._knn(noisy_points, centers, k)  # (B, N, k)

        # Gather neighbor positions and predictions
        idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbor_centers = torch.gather(
            centers.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_exp
        )
        neighbor_preds = torch.gather(
            center_pred.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_exp
        )

        # Compute features
        rel_pos = noisy_points.unsqueeze(2) - neighbor_centers  # (B, N, k, 3)
        center_disp = neighbor_preds - neighbor_centers  # (B, N, k, 3)
        dist = rel_pos.norm(dim=-1, keepdim=True)  # (B, N, k, 1)

        features = torch.cat([rel_pos, center_disp, dist], dim=-1)  # (B, N, k, 7)

        # IDW base prediction
        idw_w = 1.0 / (dist.squeeze(-1) + 1e-8)  # (B, N, k)
        idw_w = idw_w / idw_w.sum(dim=-1, keepdim=True)
        idw_base = (idw_w.unsqueeze(-1) * neighbor_preds).sum(dim=2)  # (B, N, 3)

        return features, idw_base

    def forward(
        self,
        noisy_points: Tensor,
        centers: Tensor,
        center_pred: Tensor,
    ) -> Tensor:
        """Decode dense point cloud from center predictions.

        Args:
            noisy_points: Input noisy point cloud (B, N, 3).
            centers: Center positions (B, M, 3).
            center_pred: Predicted clean centers (B, M, 3).

        Returns:
            Dense predicted point cloud (B, N, 3).
        """
        B, N, _ = noisy_points.shape
        scale_features = []
        idw_base = None

        for k, scale_mlp in zip(self.k_neighbors, self.scale_mlps):
            features, base = self._gather_neighbors(
                noisy_points, centers, center_pred, k
            )

            if idw_base is None:
                idw_base = base  # Use finest scale for base

            # Aggregate within scale
            BNk_feat = features.reshape(B * N, features.shape[2], -1)
            # Max-pool over neighbors then apply MLP
            pooled = BNk_feat.max(dim=1).values  # (B*N, 7)
            scale_out = scale_mlp(pooled)  # (B*N, hidden_dim)
            scale_features.append(scale_out)

        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=-1)  # (B*N, hidden_dim*n_scales)

        # Normalize and predict displacement
        multi_scale = self.pre_output_norm(multi_scale)
        displacement = self.fusion_mlp(multi_scale)  # (B*N, 3)
        displacement = displacement.reshape(B, N, 3)

        return idw_base + displacement
