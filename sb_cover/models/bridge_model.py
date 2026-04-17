"""BridgeFlowModel: Full-resolution bridge model with decoder refinement.

Architecture:
  1. Backbone (PVCNN2Unet): processes full N-point cloud → coarse pred (B, 3, N)
  2. FPS: extract M center predictions from coarse output
  3. PointFlowDecoder: refine dense prediction using multi-scale k-NN
     from center predictions → refined dense pred (B, N, 3)

The decoder adds local geometric reasoning that the backbone's global
U-Net architecture cannot capture, enabling the IGV geometric losses
to directly influence the decoder's learned features.
"""

from typing import Dict, Optional

import pointnet2_batch_cuda
import torch
import torch.nn as nn
from torch import Tensor

from sb_cover.models.decoder.point_flow_decoder import PointFlowDecoder


class BridgeFlowModel(nn.Module):
    """Bridge flow model with backbone + decoder refinement.

    Args:
        backbone: PVCNN2Unet model.
        cfg: Model configuration dict.
    """

    def __init__(self, backbone: nn.Module, cfg: Dict):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        # SB objective mode: 'pred_x0', 'pred_noise', or 'flow' (legacy)
        self.sb_objective = getattr(cfg, "sb_objective", "flow")

        # Number of centers for FPS
        self.num_centers = cfg.model.get("num_centers", 128)

        # PointFlowDecoder for refined dense reconstruction
        decoder_cfg = cfg.model.get("decoder", {})
        pf_cfg = decoder_cfg.get("point_flow", {})

        self.point_flow_decoder = PointFlowDecoder(
            hidden_dim=pf_cfg.get("hidden_dim", 384),
            k_neighbors=pf_cfg.get("k_neighbors", [4, 8, 16]),
            num_centers=self.num_centers,
            center_feat_dim=3,
        )

        # Register std_fwd as buffer for pred_noise mode (set by trainer)
        self.register_buffer(
            "std_fwd_lookup",
            torch.zeros(1000),
            persistent=False,
        )

    def set_std_fwd(self, std_fwd: Tensor):
        """Set the forward std lookup table from SB schedule."""
        self.std_fwd_lookup = std_fwd

    def _fps_subsample(self, points: Tensor, num_centers: int) -> Tensor:
        """CUDA FPS on (B, 3, N) input → (B, M) indices."""
        B, C, N = points.shape
        pts = points.float().transpose(1, 2).contiguous()  # (B, N, 3)
        output = torch.cuda.IntTensor(B, num_centers)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        pointnet2_batch_cuda.furthest_point_sampling_wrapper(
            B, N, num_centers, pts, temp, output
        )
        return output.long()

    def _gather(self, points: Tensor, idx: Tensor) -> Tensor:
        """Gather (B, 3, N) by idx (B, M) → (B, 3, M)."""
        idx_exp = idx.unsqueeze(1).expand(-1, points.shape[1], -1)
        return torch.gather(points, 2, idx_exp)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        x_cond: Optional[Tensor] = None,
        noise_level: Optional[Tensor] = None,
        noisy_input: Optional[Tensor] = None,
        steps: Optional[Tensor] = None,
        skip_decoder: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass: backbone → center extraction → optional decoder.

        Args:
            x: Interpolated point cloud xt (B, 3, N).
            t: Continuous time values (B,) for legacy flow matching.
            x_cond: Optional conditioning features (B, C, N).
            noise_level: Discrete SB noise levels (B,) for time embedding.
            noisy_input: Original noisy input (B, 3, N) for decoder.
                Falls back to x if not provided.
            steps: Discrete timestep indices (B,) for std_fwd lookup
                in pred_noise mode.
            skip_decoder: If True, skip PointFlowDecoder and return
                dense_pred=None. Saves ~35% compute when decoder output
                is not used at inference.

        Returns:
            Dictionary with:
                - 'velocity': Raw backbone output (B, 3, N).
                - 'pred_x0_coarse': Backbone's coarse prediction (B, 3, N).
                - 'dense_pred': Decoder-refined prediction (B, N, 3), or None.
                - 'center_idx': FPS indices (B, M).
        """
        result = {}
        time_input = noise_level if noise_level is not None else t

        # --- 1. Backbone: coarse full-resolution prediction ---
        velocity = self.backbone(x, time_input, x_cond=x_cond)  # (B, 3, N)
        result["velocity"] = velocity

        # Interpret backbone output based on objective
        if self.sb_objective == "pred_x0":
            coarse_pred = velocity
        elif self.sb_objective == "pred_noise":
            if steps is not None:
                std_fwd = self.std_fwd_lookup[steps]
                while std_fwd.ndim < velocity.ndim:
                    std_fwd = std_fwd.unsqueeze(-1)
                coarse_pred = x - std_fwd * velocity
            else:
                coarse_pred = x - velocity
        else:
            t_expand = t
            while t_expand.ndim < velocity.ndim:
                t_expand = t_expand.unsqueeze(-1)
            coarse_pred = x + (1.0 - t_expand) * velocity

        result["pred_x0_coarse"] = coarse_pred

        # --- 2. FPS: extract centers from coarse prediction ---
        center_idx = self._fps_subsample(x, self.num_centers)
        result["center_idx"] = center_idx

        centers_input = self._gather(x, center_idx)  # (B, 3, M) — xt centers
        centers_pred = self._gather(coarse_pred, center_idx)  # (B, 3, M)

        # --- 3. Decoder: refine dense prediction from center predictions ---
        if skip_decoder:
            result["dense_pred"] = None
        else:
            ref_pts = noisy_input if noisy_input is not None else x
            noisy_xyz = ref_pts.transpose(1, 2)  # (B, N, 3)
            centers_xyz = centers_input.transpose(1, 2)  # (B, M, 3)
            centers_pred_xyz = centers_pred.transpose(1, 2)  # (B, M, 3)

            dense_pred = self.point_flow_decoder(
                noisy_xyz, centers_xyz, centers_pred_xyz
            )  # (B, N, 3)
            result["dense_pred"] = dense_pred

        return result
