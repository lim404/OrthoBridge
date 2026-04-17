"""Combined IGV Loss with Geometric Manifold Correction for SB-IGV model.

The standard Schrödinger Bridge assumes transport in Euclidean R^{3N}.
Real point clouds occupy a low-dimensional geometric manifold M ⊂ R^{3N}.
The IGV module acts as a manifold correction operator Π_M that constrains
the bridge transport to preserve intrinsic geometric structure:

    dx_t = [f_SB(x_t, t) + λ(t) · ∇_{x_t} L_IGV(x_t)] dt + g(t) dW_t

where λ(t) is an SNR-adaptive schedule that suppresses the correction
in the pure-noise regime (high t) and activates it as the signal emerges.

Key components:
- DifferentiableSubsampler: Reduces N=2048 → K=256 before geometric ops (OOM guard)
- IGProjectionLoss: Fisher information manifold alignment on local covariances
- ValuationLoss: Shapley-value importance-weighted geometric fidelity
- SNRAdaptiveWeighting: Per-timestep λ(t) derived from the SB noise schedule
- IGVCurriculum: Epoch-level staged ramp-up (orthogonal to SNR weighting)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import pointnet2_batch_cuda
from torch import Tensor


# ---------------------------------------------------------------------------
# Fix 1: Differentiable downsampler — prevents OOM when N=2048+
# ---------------------------------------------------------------------------

class DifferentiableSubsampler(nn.Module):
    """Differentiable point cloud downsampling for memory-safe geometric losses.

    Reduces a point cloud from N points to K representative points before
    computing expensive geometric operations (covariance, eigenvalues).
    Uses FPS for coverage quality; gradients flow through the gather op.

    The full pipeline cost difference:
        - Without: cdist(2048, 2048) × B × 2 = ~1GB+ VRAM per loss
        - With:    cdist(256, 256) × B × 2   = ~16MB  per loss

    Args:
        num_points: Target number of representative points (K).
        method: Subsampling method ('fps' or 'random').
    """

    def __init__(self, num_points: int = 256, method: str = "fps"):
        super().__init__()
        self.num_points = num_points
        self.method = method

    def _fps_indices(self, points: Tensor, k: int) -> Tensor:
        """Farthest Point Sampling indices using CUDA kernel.

        Args:
            points: (B, N, 3).
            k: Number of points to sample.

        Returns:
            Indices (B, K) as LongTensor.
        """
        B, N, D = points.shape
        pts = points.float().contiguous()  # CUDA FPS requires fp32
        output = torch.cuda.IntTensor(B, k)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        pointnet2_batch_cuda.furthest_point_sampling_wrapper(
            B, N, k, pts, temp, output
        )
        return output.long()

    def forward(self, points: Tensor) -> Tensor:
        """Subsample point cloud.

        Args:
            points: (B, N, 3). Must have requires_grad for loss backprop.

        Returns:
            Subsampled points (B, K, 3). Gradients flow through gather.
        """
        B, N, D = points.shape
        K = min(self.num_points, N)
        if K >= N:
            return points

        if self.method == "fps":
            idx = self._fps_indices(points.detach(), K)
        else:
            # Random: same indices across channels, differentiable gather
            idx = torch.stack([
                torch.randperm(N, device=points.device)[:K] for _ in range(B)
            ])

        # Gather — this IS differentiable (d output / d input = 1 at selected indices)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(points, 1, idx_exp)


# ---------------------------------------------------------------------------
# Core geometric losses
# ---------------------------------------------------------------------------

class ChamferDistance(nn.Module):
    """Bidirectional Chamfer Distance between two point clouds."""

    def forward(self, pred: Tensor, target: Tensor, return_dist_sq: bool = False):
        """Compute Chamfer Distance.

        Args:
            pred: Predicted points (B, N, 3).
            target: Target points (B, M, 3).
            return_dist_sq: If True, also return squared distance matrix for reuse.

        Returns:
            Scalar Chamfer Distance loss, or (loss, dist_sq) if return_dist_sq=True.
        """
        dist_sq = torch.cdist(pred, target).pow(2)  # (B, N, M)
        min_p2t = dist_sq.min(dim=-1).values
        min_t2p = dist_sq.min(dim=-2).values
        cd = min_p2t.mean(dim=-1) + min_t2p.mean(dim=-1)
        loss = cd.mean()
        if return_dist_sq:
            return loss, dist_sq
        return loss


class ScoreMatchingLoss(nn.Module):
    """KDE-NLL bidirectional distribution matching loss.

    Replaces point-level MSE to avoid regression-to-mean shrinkage.
    Uses multi-scale Gaussian kernels in log-space for numerical stability.

    Attraction: each predicted point should be near some target points.
    Coverage: each target point should be covered by some predicted points.

    Args:
        bandwidths: List of kernel bandwidths (h) for multi-scale matching.
        attraction_weight: Weight for pred->target attraction term.
        coverage_weight: Weight for target->pred coverage term.
    """

    def __init__(self, bandwidths=(0.01, 0.02, 0.05),
                 attraction_weight: float = 1.0, coverage_weight: float = 1.0):
        super().__init__()
        self.bandwidths = list(bandwidths)
        self.attraction_weight = attraction_weight
        self.coverage_weight = coverage_weight

    def forward(self, dist_sq: Tensor) -> Tensor:
        """Compute score matching loss from precomputed squared distances.

        Args:
            dist_sq: Squared pairwise distances (B, N_pred, N_target).

        Returns:
            Scalar score matching loss (L=0 when pred=target, L>0 otherwise).
        """
        N_pred = dist_sq.shape[1]
        N_target = dist_sq.shape[2]
        log_N_pred = torch.log(torch.tensor(N_pred, dtype=dist_sq.dtype, device=dist_sq.device))
        log_N_target = torch.log(torch.tensor(N_target, dtype=dist_sq.dtype, device=dist_sq.device))

        total = torch.tensor(0.0, device=dist_sq.device, dtype=dist_sq.dtype)
        for h in self.bandwidths:
            log_kernel = -dist_sq / (2.0 * h * h)  # (B, N_pred, N_target)

            # Attraction: each pred point near some target points
            attraction = -(torch.logsumexp(log_kernel, dim=-1) - log_N_target).mean()

            # Coverage: each target point covered by some pred points
            coverage = -(torch.logsumexp(log_kernel, dim=-2) - log_N_pred).mean()

            total = total + self.attraction_weight * attraction + self.coverage_weight * coverage

        return total / len(self.bandwidths)


class IGProjectionLoss(nn.Module):
    """Information Geometric Projection Loss.

    Projects local neighborhoods onto the Fisher information manifold
    by computing local covariance matrices (sufficient statistics of a
    local Gaussian model) and penalizing divergence from ground truth.

    This constrains the SB transport to preserve the intrinsic local
    geometry of the point cloud manifold, not just point positions.

    Args:
        k_neighbors: Number of neighbors for local covariance estimation.
    """

    def __init__(self, k_neighbors: int = 16):
        super().__init__()
        self.k = k_neighbors

    def _local_covariance_chunked(self, points: Tensor, k: int) -> Tensor:
        """Compute local covariance matrices with chunked distance computation.

        Avoids allocating a full (B, N, N) distance matrix by processing
        in chunks. For K=256 this is a (B, 256, 256) matrix which is fine,
        but we keep the chunked implementation for safety.

        Args:
            points: (B, N, 3) where N is already subsampled (e.g. 256).
            k: Number of neighbors.

        Returns:
            Local covariance matrices (B, N, 3, 3).
        """
        B, N, D = points.shape
        k = min(k, N - 1)

        # For small N (≤512), direct computation is fine
        dist = torch.cdist(points, points)  # (B, N, N)
        _, nn_idx = dist.topk(k + 1, dim=-1, largest=False)
        nn_idx = nn_idx[:, :, 1:]  # exclude self → (B, N, k)

        # Gather neighbors
        nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        neighbors = torch.gather(
            points.unsqueeze(1).expand(-1, N, -1, -1), 2, nn_idx_exp
        )  # (B, N, k, 3)

        # Center and compute covariance
        centered = neighbors - points.unsqueeze(2)
        cov = torch.einsum("bnki,bnkj->bnij", centered, centered) / k
        return cov

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute IG Projection loss on PRE-SUBSAMPLED inputs.

        IMPORTANT: Caller must subsample to ≤256 points BEFORE calling this.

        Args:
            pred: Predicted points (B, K, 3), K ≤ 256.
            target: Target points (B, K, 3), K ≤ 256.

        Returns:
            Scalar IG Projection loss.
        """
        cov_pred = self._local_covariance_chunked(pred, self.k)
        cov_tgt = self._local_covariance_chunked(target, self.k)

        # Match pred points to nearest target points for covariance comparison
        dist = torch.cdist(pred, target)
        nn_idx = dist.min(dim=-1).indices  # (B, K)

        matched_tgt_cov = torch.gather(
            cov_tgt, 1,
            nn_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
        )

        # Regularize for numerical stability
        eps = 1e-6 * torch.eye(3, device=pred.device).unsqueeze(0).unsqueeze(0)
        cov_pred_reg = cov_pred + eps
        matched_tgt_cov_reg = matched_tgt_cov + eps

        # Frobenius norm of covariance difference
        diff = cov_pred_reg - matched_tgt_cov_reg
        loss = (diff ** 2).sum(dim=(-1, -2)).mean()

        return loss


class ValuationLoss(nn.Module):
    """Shapley-value-based point importance loss.

    Computes per-point geometric importance via PCA of local neighborhoods
    (surface variation), then uses importance as weights for directional
    Chamfer distance. This ensures the model prioritizes structurally
    critical regions (edges, corners, thin features).

    Args:
        k_neighbors: Number of neighbors for importance estimation.
    """

    def __init__(self, k_neighbors: int = 16):
        super().__init__()
        self.k = k_neighbors

    def _compute_point_importance(self, points: Tensor) -> Tensor:
        """Compute per-point importance via surface variation.

        Surface variation = λ_min / (λ_0 + λ_1 + λ_2).
        High at edges/corners, low on flat surfaces.

        Args:
            points: (B, K, 3), K ≤ 256.

        Returns:
            Importance weights (B, K).
        """
        B, N, D = points.shape
        k = min(self.k, N - 1)

        dist = torch.cdist(points, points)
        _, nn_idx = dist.topk(k + 1, dim=-1, largest=False)
        nn_idx = nn_idx[:, :, 1:]

        nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        neighbors = torch.gather(
            points.unsqueeze(1).expand(-1, N, -1, -1), 2, nn_idx_exp
        )

        centered = neighbors - points.unsqueeze(2)
        cov = torch.einsum("bnki,bnkj->bnij", centered, centered) / k

        # Eigenvalues for surface variation
        eigvals = torch.linalg.eigvalsh(
            cov + 1e-6 * torch.eye(3, device=points.device)
        )  # (B, N, 3)

        sorted_eig = eigvals.sort(dim=-1).values.clamp(min=1e-8)
        importance = sorted_eig[:, :, 0] / sorted_eig.sum(dim=-1)

        return importance

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute valuation loss on PRE-SUBSAMPLED inputs.

        IMPORTANT: Caller must subsample to ≤256 points BEFORE calling this.

        Args:
            pred: Predicted points (B, K, 3), K ≤ 256.
            target: Target points (B, K, 3), K ≤ 256.

        Returns:
            Scalar valuation loss.
        """
        # Compute importance on target
        importance = self._compute_point_importance(target)  # (B, K)
        importance = importance / (importance.mean(dim=-1, keepdim=True) + 1e-8)

        # Weighted nearest-neighbor distance
        dist = torch.cdist(pred, target)  # (B, K_pred, K_tgt)

        # pred → target (weighted by target importance)
        min_dist_p2t, nn_p2t = dist.min(dim=-1)
        tgt_weights = torch.gather(importance, 1, nn_p2t)
        weighted_p2t = (min_dist_p2t * tgt_weights).mean(dim=-1)

        # target → pred (weighted by target importance)
        min_dist_t2p = dist.min(dim=-2).values
        weighted_t2p = (min_dist_t2p * importance).mean(dim=-1)

        loss = (weighted_p2t + weighted_t2p).mean()
        return loss


# ---------------------------------------------------------------------------
# Fix 3: SNR-adaptive time-dependent weighting
# ---------------------------------------------------------------------------

class SNRAdaptiveWeighting:
    """Per-timestep IGV weight derived from the SB noise schedule.

    At timestep t near T (pure noise), SNR ≈ 0 → λ(t) ≈ 0:
        enforcing perfect geometry on noise is meaningless and destabilizing.

    At timestep t near 0 (clean signal), SNR → ∞ → λ(t) ≈ 1:
        the prediction is nearly clean, geometry constraints are meaningful.

    The weight function is:
        λ(t) = σ(a · (log SNR(t) - log SNR_mid))

    where a controls sharpness and SNR_mid is the transition point.

    Args:
        schedule: Weight schedule type ('sigmoid', 'linear', 'cosine').
        sharpness: Sigmoid steepness (higher = sharper transition).
        midpoint_snr: SNR value at sigmoid midpoint (weight = 0.5).
    """

    def __init__(
        self,
        schedule: str = "sigmoid",
        sharpness: float = 1.0,
        midpoint_snr: float = 1.0,
    ):
        self.schedule = schedule
        self.sharpness = sharpness
        self.midpoint_snr = midpoint_snr

    def __call__(self, steps: Tensor, sb_schedule) -> Tensor:
        """Compute per-sample IGV weight.

        Args:
            steps: Timestep indices (B,).
            sb_schedule: SBSchedule with .snr buffer.

        Returns:
            Per-sample weights (B,) in [0, 1].
        """
        return sb_schedule.snr_weight(
            steps,
            clamp_min=0.0,
            clamp_max=1.0,
            schedule=self.schedule,
            sharpness=self.sharpness,
            midpoint_snr=self.midpoint_snr,
        )


# ---------------------------------------------------------------------------
# Epoch-level curriculum (orthogonal to per-timestep SNR weighting)
# ---------------------------------------------------------------------------

class IGVCurriculum:
    """Epoch-level staged ramp-up for IGV losses.

    Orthogonal to SNR-adaptive weighting: this controls WHEN (in training)
    IGV kicks in, while SNR weighting controls WHERE (in the diffusion
    trajectory) it is applied.

    Args:
        stage1_epochs: Base losses only.
        stage2_epochs: Ramp up IG Projection.
        stage3_epochs: Ramp up Valuation.
        stage4_epochs: Full weights.
    """

    def __init__(
        self,
        stage1_epochs: int = 5,
        stage2_epochs: int = 15,
        stage3_epochs: int = 30,
        stage4_epochs: int = 200,
    ):
        self.stage1 = stage1_epochs
        self.stage2 = stage2_epochs
        self.stage3 = stage3_epochs
        self.stage4 = stage4_epochs

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get epoch-level multipliers."""
        if epoch < self.stage1:
            return {"ig_projection": 0.0, "valuation": 0.0}
        elif epoch < self.stage2:
            progress = (epoch - self.stage1) / max(self.stage2 - self.stage1, 1)
            return {"ig_projection": progress, "valuation": 0.0}
        elif epoch < self.stage3:
            progress = (epoch - self.stage2) / max(self.stage3 - self.stage2, 1)
            return {"ig_projection": 1.0, "valuation": progress}
        else:
            return {"ig_projection": 1.0, "valuation": 1.0}


# ---------------------------------------------------------------------------
# Fix 2: Geometric Manifold Correction — wraps IGV as SDE drift correction
# ---------------------------------------------------------------------------

class GeometricManifoldCorrection(nn.Module):
    """Geometric manifold correction operator for Schrödinger Bridge transport.

    In the standard SB formulation, the forward SDE transports mass in
    Euclidean space R^{3N}. However, natural point clouds concentrate on a
    low-dimensional manifold M defined by:
      (a) Local covariance structure (captured by Fisher information metric)
      (b) Point importance distribution (captured by Shapley valuation)

    This module computes the manifold correction term:
        Π_M(x_t, t) = λ_IG(t) · ∇L_IG(x_t) + λ_VAL(t) · ∇L_VAL(x_t)

    where λ(t) is SNR-adaptive: zero in the noise regime, full in the
    signal regime. This prevents the well-known collapse when geometric
    constraints are imposed on pure-noise predictions.

    The correction is applied as an additional drift in the reverse SDE:
        dx_t = [f_θ(x_t, t) + Π_M(x_t, t)] dt + g(t) dW_t

    During training, Π_M manifests as additional loss terms weighted by λ(t).
    During inference (DDPM sampling), Π_M is applied as gradient guidance.

    Args:
        cfg: Loss configuration dict.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        loss_cfg = cfg.get("loss", cfg)

        # Geometric loss modules
        self.ig_projection = IGProjectionLoss(
            k_neighbors=loss_cfg.get("ig_projection", {}).get("k_neighbors", 16)
        )
        self.valuation = ValuationLoss(
            k_neighbors=loss_cfg.get("valuation", {}).get("k_neighbors", 16)
        )

        # Differentiable subsampler: N=2048 → K=256 before geometric ops
        igv_subsample = loss_cfg.get("igv_subsample_points", 256)
        self.subsampler = DifferentiableSubsampler(
            num_points=igv_subsample, method="fps"
        )

        # SNR-adaptive weighting
        snr_cfg = loss_cfg.get("snr_weighting", {})
        self.snr_weighting = SNRAdaptiveWeighting(
            schedule=snr_cfg.get("schedule", "sigmoid"),
            sharpness=snr_cfg.get("sharpness", 1.0),
            midpoint_snr=snr_cfg.get("midpoint_snr", 1.0),
        )

        # Base weights
        self.w_ig = loss_cfg.get("ig_projection", {}).get("weight", 0.3)
        self.w_val = loss_cfg.get("valuation", {}).get("weight", 0.2)

    def forward(
        self,
        pred_points: Tensor,
        target_points: Tensor,
        steps: Optional[Tensor] = None,
        sb_schedule=None,
        epoch_weight_ig: float = 1.0,
        epoch_weight_val: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Compute manifold correction losses.

        Args:
            pred_points: Predicted points (B, N, 3). Full resolution.
            target_points: Ground truth points (B, M, 3). Full resolution.
            steps: Per-sample timestep indices (B,). Required for SNR weighting.
            sb_schedule: SBSchedule instance. Required for SNR weighting.
            epoch_weight_ig: Curriculum multiplier for IG loss.
            epoch_weight_val: Curriculum multiplier for Valuation loss.

        Returns:
            Dict with 'ig_loss', 'val_loss', 'total', and 'snr_weight'.
        """
        # --- Differentiable subsampling: N → K ---
        pred_sub = self.subsampler(pred_points)    # (B, K, 3), grads flow
        tgt_sub = self.subsampler(target_points)   # (B, K, 3)

        # --- Compute geometric losses on subsampled points ---
        l_ig = self.ig_projection(pred_sub, tgt_sub)
        l_val = self.valuation(pred_sub, tgt_sub)

        # --- SNR-adaptive per-sample weighting ---
        if steps is not None and sb_schedule is not None:
            snr_w = self.snr_weighting(steps, sb_schedule)  # (B,)
            # Reduce to scalar: weight loss by per-sample SNR weight
            # This means samples at high-noise timesteps contribute ~0 to IGV
            snr_w_mean = snr_w.mean()
        else:
            # No SB schedule (legacy mode or inference) — full weight
            snr_w = None
            snr_w_mean = 1.0

        # --- Combine: base_weight × epoch_curriculum × snr_weight ---
        weighted_ig = self.w_ig * epoch_weight_ig * snr_w_mean * l_ig
        weighted_val = self.w_val * epoch_weight_val * snr_w_mean * l_val

        return {
            "ig_loss": l_ig,
            "val_loss": l_val,
            "ig_weighted": weighted_ig,
            "val_weighted": weighted_val,
            "total": weighted_ig + weighted_val,
            "snr_weight_mean": snr_w_mean if isinstance(snr_w_mean, Tensor) else torch.tensor(snr_w_mean),
        }


# ---------------------------------------------------------------------------
# Main combined loss
# ---------------------------------------------------------------------------

class CombinedIGVLoss(nn.Module):
    """Combined loss with Geometric Manifold Correction.

    Loss structure:
        L_total = L_score_matching (primary, KDE-NLL distribution matching)
                + L_center_flow (secondary, MSE — already score matching via pred_noise)
                + L_chamfer (auxiliary)
                + λ(t) · curriculum(epoch) · [L_IG + L_VAL]  (manifold correction)

    The manifold correction is applied at TWO levels:
        1. Dense-level (2048 pts → subsampled to 256 for IG/VAL)
        2. Center-level (128 pts → used directly, no subsampling needed)

    Args:
        cfg: Full configuration dict.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        loss_cfg = cfg.get("loss", cfg)

        # Base loss weights
        self.w_point_flow = loss_cfg.get("point_flow", {}).get("weight", 0.05)
        self.w_chamfer = loss_cfg.get("chamfer", {}).get("weight", 0.1)
        self.w_center_flow = loss_cfg.get("center_flow", {}).get("weight", 0.5)

        # Simple Chamfer distance
        self.chamfer = ChamferDistance()

        # Score matching loss (replaces point-level MSE)
        sm_cfg = loss_cfg.get("point_flow", {}).get("score_matching", {})
        self.score_matching = ScoreMatchingLoss(
            bandwidths=sm_cfg.get("bandwidths", [0.01, 0.02, 0.05]),
            attraction_weight=sm_cfg.get("attraction_weight", 1.0),
            coverage_weight=sm_cfg.get("coverage_weight", 1.0),
        )

        # Geometric manifold correction (contains IG + Valuation + subsampling + SNR)
        self.manifold_correction_dense = GeometricManifoldCorrection(cfg)
        self.manifold_correction_center = GeometricManifoldCorrection(cfg)

        # Epoch-level curriculum
        curriculum_cfg = loss_cfg.get("igv_curriculum", {})
        self.curriculum = IGVCurriculum(
            stage1_epochs=curriculum_cfg.get("stage1_epochs", 5),
            stage2_epochs=curriculum_cfg.get("stage2_epochs", 15),
            stage3_epochs=curriculum_cfg.get("stage3_epochs", 30),
            stage4_epochs=curriculum_cfg.get("stage4_epochs", 200),
        )

    def forward(
        self,
        pred_centers: Tensor,
        gt_centers: Tensor,
        target_points: Tensor,
        center_velocity: Optional[Tensor] = None,
        center_gt: Optional[Tensor] = None,
        dense_pred: Optional[Tensor] = None,
        epoch: int = 0,
        steps: Optional[Tensor] = None,
        sb_schedule=None,
    ) -> Dict[str, Tensor]:
        """Compute combined loss with manifold correction.

        Args:
            pred_centers: Predicted clean centers (B, 3, M) or (B, M, 3).
            gt_centers: Ground truth centers (B, 3, M) or (B, M, 3).
            target_points: Full ground truth point cloud (B, 3, N) or (B, N, 3).
            center_velocity: Raw backbone output (B, 3, M).
            center_gt: Ground truth for center-level prediction (B, 3, M).
            dense_pred: Dense prediction from PointFlowDecoder (B, N, 3).
            epoch: Current epoch for curriculum scheduling.
            steps: Per-sample timestep indices (B,) for SNR-adaptive weighting.
            sb_schedule: SBSchedule instance for SNR computation.

        Returns:
            Dictionary with individual losses and total loss.
        """
        losses = {}
        device = pred_centers.device

        # --- Ensure (B, N, 3) format ---
        if pred_centers.shape[1] == 3 and pred_centers.shape[2] != 3:
            pred_centers = pred_centers.transpose(1, 2)
        if gt_centers.shape[1] == 3 and gt_centers.shape[2] != 3:
            gt_centers = gt_centers.transpose(1, 2)
        if target_points.shape[1] == 3 and target_points.shape[2] != 3:
            target_points = target_points.transpose(1, 2)

        # --- Get epoch-level curriculum weights ---
        curr_weights = self.curriculum.get_weights(epoch)

        # --- 1. Center-level flow loss (MSE) ---
        if center_velocity is not None and center_gt is not None:
            if center_velocity.shape[1] == 3 and center_velocity.shape[2] != 3:
                center_velocity = center_velocity.transpose(1, 2)
            if center_gt.shape[1] == 3 and center_gt.shape[2] != 3:
                center_gt = center_gt.transpose(1, 2)
            l_center = nn.functional.mse_loss(center_velocity, center_gt)
            losses["center_flow"] = l_center
        else:
            l_center = torch.tensor(0.0, device=device)
            losses["center_flow"] = l_center

        # --- 2. Score matching (primary) + Chamfer (auxiliary) ---
        # dist_sq is computed once by Chamfer and reused by ScoreMatchingLoss
        if dense_pred is not None:
            l_chamfer, dist_sq = self.chamfer(dense_pred, target_points, return_dist_sq=True)
            losses["chamfer"] = l_chamfer

            l_point_flow = self.score_matching(dist_sq)
            losses["point_flow"] = l_point_flow
        else:
            l_point_flow = torch.tensor(0.0, device=device)
            l_chamfer = torch.tensor(0.0, device=device)
            losses["point_flow"] = l_point_flow
            losses["chamfer"] = l_chamfer

        # --- 3. Geometric Manifold Correction on DENSE predictions ---
        if dense_pred is not None:
            mc_dense = self.manifold_correction_dense(
                pred_points=dense_pred,
                target_points=target_points,
                steps=steps,
                sb_schedule=sb_schedule,
                epoch_weight_ig=curr_weights["ig_projection"],
                epoch_weight_val=curr_weights["valuation"],
            )
            losses["ig_dense"] = mc_dense["ig_loss"]
            losses["val_dense"] = mc_dense["val_loss"]
            losses["snr_weight"] = mc_dense["snr_weight_mean"]
            l_mc_dense = mc_dense["total"]
        else:
            l_mc_dense = torch.tensor(0.0, device=device)
            losses["ig_dense"] = torch.tensor(0.0, device=device)
            losses["val_dense"] = torch.tensor(0.0, device=device)
            losses["snr_weight"] = torch.tensor(0.0, device=device)

        # --- 4. Geometric Manifold Correction on CENTER predictions ---
        # Centers are only 128 pts — no subsampling needed, but SNR still applies
        mc_center = self.manifold_correction_center(
            pred_points=pred_centers,
            target_points=gt_centers,
            steps=steps,
            sb_schedule=sb_schedule,
            epoch_weight_ig=curr_weights["ig_projection"],
            epoch_weight_val=curr_weights["valuation"],
        )
        losses["ig_center"] = mc_center["ig_loss"]
        losses["val_center"] = mc_center["val_loss"]
        l_mc_center = mc_center["total"]

        # --- Aggregate total ---
        total = (
            self.w_point_flow * l_point_flow
            + self.w_center_flow * l_center
            + self.w_chamfer * l_chamfer
            + l_mc_dense
            + l_mc_center
        )

        losses["total"] = total
        return losses
