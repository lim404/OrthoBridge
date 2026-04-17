"""Local decoder for propagating center predictions to dense point clouds.

Uses k-nearest neighbor grouping to propagate center-level predictions
to all input points via local feature aggregation.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ResidualMLP(nn.Module):
    """MLP block with residual connection.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output feature dimension.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features (B, N, C) or (B*N, C).

        Returns:
            Output features with residual connection.
        """
        return self.net(x) + self.shortcut(x)


class LocalDecoder(nn.Module):
    """Local neighborhood decoder for dense point prediction.

    Groups nearby points around each center and uses local feature
    aggregation to produce per-point displacements from center predictions.

    Args:
        num_neighbors: Number of nearest neighbors per center.
        center_dim: Dimension of center features (3 for xyz).
        hidden_dim: Hidden dimension for MLP layers.
        num_centers: Expected number of centers.
    """

    def __init__(
        self,
        num_neighbors: int = 32,
        center_dim: int = 3,
        hidden_dim: int = 128,
        num_centers: int = 128,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.num_centers = num_centers

        # Input: relative position (3) + center feature (center_dim) + distance (1)
        in_dim = 3 + center_dim + 1

        self.mlp = nn.Sequential(
            ResidualMLP(in_dim, hidden_dim, hidden_dim),
            ResidualMLP(hidden_dim, hidden_dim, hidden_dim),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def _knn(self, query: Tensor, reference: Tensor, k: int) -> Tensor:
        """K-nearest neighbor search.

        Args:
            query: Query points (B, N, 3).
            reference: Reference points (B, M, 3).
            k: Number of neighbors.

        Returns:
            Indices of k nearest neighbors (B, N, k).
        """
        # (B, N, 1, 3) - (B, 1, M, 3) -> (B, N, M)
        dist = torch.cdist(query, reference)
        _, idx = dist.topk(k, dim=-1, largest=False)
        return idx

    def forward(
        self,
        noisy_points: Tensor,
        centers: Tensor,
        center_pred: Tensor,
    ) -> Tensor:
        """Propagate center predictions to dense point cloud.

        Args:
            noisy_points: Input noisy points (B, N, 3).
            centers: Center positions (B, M, 3).
            center_pred: Predicted clean center positions (B, M, 3).

        Returns:
            Dense predicted points (B, N, 3).
        """
        B, N, _ = noisy_points.shape
        M = centers.shape[1]
        k = min(self.num_neighbors, M)

        # Find k nearest centers for each point
        nn_idx = self._knn(noisy_points, centers, k)  # (B, N, k)

        # Gather center predictions for neighbors
        nn_idx_expanded = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbor_centers = torch.gather(
            centers.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            nn_idx_expanded,
        )  # (B, N, k, 3)
        neighbor_preds = torch.gather(
            center_pred.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            nn_idx_expanded,
        )  # (B, N, k, 3)

        # Relative positions
        rel_pos = noisy_points.unsqueeze(2) - neighbor_centers  # (B, N, k, 3)
        dist = rel_pos.norm(dim=-1, keepdim=True)  # (B, N, k, 1)

        # Center displacement feature
        center_disp = neighbor_preds - neighbor_centers  # (B, N, k, 3)

        # Concatenate features
        features = torch.cat([rel_pos, center_disp, dist], dim=-1)  # (B, N, k, 3+3+1)

        # Process through MLP
        BN = B * N
        features = features.reshape(BN * k, -1)
        features = self.mlp(features)  # (B*N*k, hidden)
        features = features.reshape(BN, k, -1)

        # Aggregate neighbors (inverse-distance weighted mean)
        weights = 1.0 / (dist.reshape(BN, k, 1) + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        features = (features * weights).sum(dim=1)  # (B*N, hidden)

        # Output displacement
        displacement = self.output_mlp(features)  # (B*N, 3)
        displacement = displacement.reshape(B, N, 3)

        # Apply displacement to IDW-interpolated center predictions
        # IDW interpolation of center predictions to each point
        nn_dist = torch.cdist(noisy_points, centers)  # (B, N, M)
        nn_dist_k = torch.gather(nn_dist, 2, nn_idx)  # (B, N, k)
        idw_weights = 1.0 / (nn_dist_k + 1e-8)
        idw_weights = idw_weights / idw_weights.sum(dim=-1, keepdim=True)  # (B, N, k)

        # Weighted sum of neighbor predictions
        base_pred = (idw_weights.unsqueeze(-1) * neighbor_preds).sum(dim=2)  # (B, N, 3)

        return base_pred + displacement
