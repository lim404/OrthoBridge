"""Generalizability evaluation across backbones and geometric objectives.

Tier 1 — Cross-backbone:
  A. PVCNN (pretrained P2P-Bridge neural denoiser)
  B. Bilateral filter (classical, k-NN weighted averaging)
  C. Local PCA projection (classical, project to fitted plane)

Tier 2 — Cross-geometric-objective (on PVCNN backbone):
  A. Repulsion + Projection uniformity (current paper loss)
  B. Laplacian smoothing (push toward neighbor centroid)
  C. Normal consistency (penalize local normal variation)

For each combination: Baseline vs Standard guidance vs Orthogonal guidance.

Usage:
    python eval_generalizability.py --num_shapes 100
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import argparse
import json
import math
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm

from sb_cover.training.trainer_igv import TrainerIGV
from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
from sb_cover.evaluation.guided_sampling import (
    _project_orthogonal, GeometricQualityLoss,
    ddpm_denoise_guided, ddpm_denoise_ortho_guided,
)
from sb_cover.losses.sb_interpolation import SBSchedule, space_indices
from sb_cover.data.punet_loader import get_punet_loaders
from models.train_utils import to_cuda
from metrics.geometric_metrics import compute_vd, compute_igsd


# ====================================================================
#  Classical denoisers (Tier 1: cross-backbone)
# ====================================================================

def bilateral_denoise(x: Tensor, iterations: int = 5, k: int = 16,
                      sigma_d: float = 0.1) -> Tensor:
    """Bilateral filter denoiser.  x: (B, 3, N) → pred_x0: (B, 3, N)."""
    pts = x.clone()
    for _ in range(iterations):
        p = pts.transpose(1, 2)  # (B, N, 3)
        dist = torch.cdist(p, p)  # (B, N, N)
        _, nn_idx = dist.topk(k + 1, dim=-1, largest=False)
        nn_idx = nn_idx[:, :, 1:]  # (B, N, k)

        # Gather neighbors
        B, N, _ = p.shape
        nn_pts = torch.gather(
            p.unsqueeze(2).expand(-1, -1, N, -1),
            2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )  # (B, N, k, 3)

        # Bilateral weights: spatial Gaussian
        nn_dist = torch.gather(dist, 2, nn_idx)  # (B, N, k)
        weights = torch.exp(-nn_dist ** 2 / (2 * sigma_d ** 2))  # (B, N, k)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Weighted average
        smoothed = (nn_pts * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, 3)
        pts = smoothed.transpose(1, 2)  # (B, 3, N)
    return pts


def pca_projection_denoise(x: Tensor, iterations: int = 3,
                           k: int = 12) -> Tensor:
    """Local PCA projection denoiser.  Projects each point onto its
    local tangent plane fitted via PCA.  x: (B, 3, N) → pred_x0.
    Processes per-sample on CPU for numerical stability."""
    B, C, N = x.shape
    result = x.clone()

    for bi in range(B):
        pts = x[bi].T.cpu().float()  # (N, 3) on CPU
        for _ in range(iterations):
            dist = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
            _, nn_idx = dist.topk(k + 1, dim=-1, largest=False)
            nn_idx = nn_idx[:, 1:]  # (N, k)

            nn_pts = pts[nn_idx]  # (N, k, 3)
            centroid = nn_pts.mean(dim=1)  # (N, 3)
            centered = nn_pts - centroid.unsqueeze(1)
            cov = torch.einsum("nki,nkj->nij", centered, centered) / k
            cov = cov + 1e-5 * torch.eye(3)

            eigvals, eigvecs = torch.linalg.eigh(cov)
            normal = eigvecs[:, :, 0]  # (N, 3)

            diff = pts - centroid
            proj_len = (diff * normal).sum(dim=-1, keepdim=True)
            pts = pts - proj_len * normal

        result[bi] = pts.T.to(x.device)
    return result


# ====================================================================
#  Generic guided iterative denoising (works with ANY denoiser)
# ====================================================================

def two_stage_guided(
    denoise_fn: Callable,
    x_noisy: Tensor,
    refine_steps: int = 10,
    guidance: str = "none",       # "none", "standard", "ortho"
    guidance_scale: float = 3.0,
    geom_loss_fn: Optional[nn.Module] = None,
    grad_clip: float = 5.0,
) -> Tensor:
    """Two-stage denoising: backbone first, then guidance refinement.

    Stage 1: Run classical denoiser to convergence (no guidance).
    Stage 2: Iterative guidance-only refinement using the denoiser's
             score direction for orthogonal projection.

    This cleanly separates backbone quality from guidance mechanism.
    """
    # Stage 1: Denoise to convergence
    x_denoised = denoise_fn(x_noisy)

    if guidance == "none" or geom_loss_fn is None:
        return x_denoised

    # Stage 2: Guidance refinement
    xt = x_denoised.clone()

    for step in range(refine_steps):
        # Re-estimate score (denoising direction) at current position
        pred_x0 = denoise_fn(xt)
        score = pred_x0 - xt  # (B, 3, N)

        # Guidance gradient
        x_g = xt.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = geom_loss_fn(x_g)["total"]
            if loss.requires_grad:
                grad = torch.autograd.grad(loss, x_g)[0]
                grad = grad.clamp(-grad_clip, grad_clip)

                if guidance == "ortho":
                    grad = _project_orthogonal(grad, score.detach())

                xt = xt - guidance_scale * grad
            else:
                break

        xt = xt.detach()

    return xt


# ====================================================================
#  Alternative geometric objectives (Tier 2)
# ====================================================================

class LaplacianSmoothingLoss(nn.Module):
    """Push each point toward centroid of its k-NN.
    Minimizing this enforces local smoothness / uniformity."""

    def __init__(self, k: int = 8, subsample_n: int = 512):
        super().__init__()
        self.k = k
        self.subsample_n = subsample_n

    def forward(self, points_b3n: Tensor) -> Dict[str, Tensor]:
        pts = points_b3n.transpose(1, 2)  # (B, N, 3)
        B, N, _ = pts.shape
        K = min(self.subsample_n, N)
        if K < N:
            idx = torch.randperm(N, device=pts.device)[:K]
            pts_sub = pts[:, idx]
        else:
            pts_sub = pts

        dist = torch.cdist(pts_sub, pts_sub)
        _, nn_idx = dist.topk(self.k + 1, dim=-1, largest=False)
        nn_idx = nn_idx[:, :, 1:]

        nn_pts = torch.gather(
            pts_sub.unsqueeze(2).expand(-1, -1, pts_sub.shape[1], -1),
            2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        centroid = nn_pts.mean(dim=2)  # (B, K, 3)
        lap_loss = ((pts_sub - centroid) ** 2).sum(dim=-1).mean()
        return {"total": lap_loss, "laplacian": lap_loss}


class KNNUniformityLoss(nn.Module):
    """Penalize non-uniform kNN distances.
    Variance of mean-kNN-distance → low = uniform spacing."""

    def __init__(self, k: int = 6, subsample_n: int = 512):
        super().__init__()
        self.k = k
        self.subsample_n = subsample_n

    def forward(self, points_b3n: Tensor) -> Dict[str, Tensor]:
        pts = points_b3n.transpose(1, 2)  # (B, N, 3)
        B, N, _ = pts.shape
        K = min(self.subsample_n, N)
        if K < N:
            idx = torch.randperm(N, device=pts.device)[:K]
            pts_sub = pts[:, idx]
        else:
            pts_sub = pts

        dist = torch.cdist(pts_sub, pts_sub)
        knn_dist, _ = dist.topk(self.k + 1, dim=-1, largest=False)
        knn_dist = knn_dist[:, :, 1:]  # (B, K, k) exclude self
        mean_knn = knn_dist.mean(dim=-1)  # (B, K)
        # Penalize variance of per-point mean kNN distances
        uniformity_loss = mean_knn.var(dim=-1).mean()
        return {"total": uniformity_loss, "knn_uniformity": uniformity_loss}


# ====================================================================
#  Metrics
# ====================================================================

def compute_metrics_batch(pred, gt):
    B = pred.shape[0]
    cds, vds, igsds = [], [], []
    for i in range(B):
        p, g = pred[i].T, gt[i].T
        d = torch.cdist(p.unsqueeze(0), g.unsqueeze(0)).squeeze(0)
        cds.append((d.min(1).values.mean() + d.min(0).values.mean()).item() * 1e3)
        vds.append(compute_vd(p, g))
        igsds.append(compute_igsd(p, g))
    return np.array(cds), np.array(vds), np.array(igsds)


# ====================================================================
#  Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shapes", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/PVDS_PUNet/latest.pth")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--out_dir", type=str,
                        default="experiments/generalizability")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = args.guidance_scale

    # ── Load neural model ──
    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone(args.checkpoint)
    trainer.model.eval()

    # ── Load test data ──
    _, test_loader = get_punet_loaders(
        data_dir=cfg.data.data_dir,
        patch_size=cfg.data.get("npoints", 2048),
        batch_size=args.batch_size,
        noise_min=args.noise_std, noise_max=args.noise_std,
        num_workers=2, num_patches=10,
    )
    all_clean, all_noisy = [], []
    n = 0
    for batch in test_loader:
        if n >= args.num_shapes:
            break
        batch = to_cuda(batch, device)
        xc, xn = trainer._extract_batch(batch)
        use = min(xc.shape[0], args.num_shapes - n)
        all_clean.append(xc[:use])
        all_noisy.append(xn[:use])
        n += use
    all_clean = torch.cat(all_clean, 0)
    all_noisy = torch.cat(all_noisy, 0)
    N = all_clean.shape[0]
    print(f"Test shapes: {N},  σ = {args.noise_std},  λ = {lam}\n")

    # ── Geometric losses ──
    geom_losses = {
        "Repulsion+Proj": GeometricQualityLoss(
            w_repulsion=1.0, w_projection=1.0, w_covariance=0.0,
            subsample_n=512),
        "Laplacian": LaplacianSmoothingLoss(k=8, subsample_n=512),
        "kNN-Uniform": KNNUniformityLoss(k=6, subsample_n=512),
    }

    all_results = []

    def record(tier, backbone, objective, guidance_name, pred):
        cd, vd, igsd = compute_metrics_batch(pred, all_clean)
        row = {
            "tier": tier, "backbone": backbone, "objective": objective,
            "guidance": guidance_name,
            "CD": cd.mean(), "CD_std": cd.std(),
            "VD": vd.mean(), "VD_std": vd.std(),
            "IGSD": igsd.mean(), "IGSD_std": igsd.std(),
        }
        all_results.append(row)
        print(f"    {guidance_name:<12s}  CD={row['CD']:.2f}  "
              f"VD={row['VD']:.4f}  IGSD={row['IGSD']:.6f}")
        return row

    # ================================================================
    #  TIER 1: Cross-backbone (fixed objective = Repulsion+Proj)
    # ================================================================
    print("=" * 70)
    print("  TIER 1: Cross-backbone generalizability")
    print("=" * 70)

    geom = geom_losses["Repulsion+Proj"]

    # ── A. PVCNN (neural) ──
    print(f"\n[Backbone A] PVCNN (P2P-Bridge, neural)")
    # Baseline
    with torch.no_grad():
        res = ddpm_denoise(trainer.model, trainer.sb_schedule, all_noisy,
                           sampling_steps=args.sampling_steps, verbose=True)
    record("T1", "PVCNN", "Repulsion+Proj", "Baseline", res["x_pred"])

    # Standard
    res = ddpm_denoise_guided(
        trainer.model, trainer.sb_schedule, all_noisy,
        sampling_steps=args.sampling_steps, guidance_scale=lam,
        annealing="constant", geom_loss=geom, grad_clip=5.0, verbose=True)
    record("T1", "PVCNN", "Repulsion+Proj", "Standard", res["x_pred"])

    # Orthogonal
    res = ddpm_denoise_ortho_guided(
        trainer.model, trainer.sb_schedule, all_noisy,
        sampling_steps=args.sampling_steps, guidance_scale=lam,
        annealing="linear_decay", geom_loss=geom, grad_clip=5.0, verbose=True)
    record("T1", "PVCNN", "Repulsion+Proj", "Ortho", res["x_pred"])

    # ── B. Bilateral filter (classical) ──
    classical_lam = lam * 5  # Classical needs stronger λ (less precise score)
    print(f"\n[Backbone B] Bilateral filter (classical, λ_guidance={classical_lam})")
    for gtype in ["Baseline", "Standard", "Ortho"]:
        guidance = {"Baseline": "none", "Standard": "standard",
                    "Ortho": "ortho"}[gtype]
        pred = two_stage_guided(
            denoise_fn=lambda x: bilateral_denoise(x, iterations=5, k=16),
            x_noisy=all_noisy, refine_steps=10,
            guidance=guidance, guidance_scale=classical_lam,
            geom_loss_fn=geom if guidance != "none" else None,
            grad_clip=5.0)
        record("T1", "Bilateral", "Repulsion+Proj", gtype, pred)

    # ── C. PCA projection (classical) ──
    print(f"\n[Backbone C] Local PCA projection (classical, λ_guidance={classical_lam})")
    for gtype in ["Baseline", "Standard", "Ortho"]:
        guidance = {"Baseline": "none", "Standard": "standard",
                    "Ortho": "ortho"}[gtype]
        pred = two_stage_guided(
            denoise_fn=lambda x: pca_projection_denoise(x, iterations=3, k=12),
            x_noisy=all_noisy, refine_steps=10,
            guidance=guidance, guidance_scale=classical_lam,
            geom_loss_fn=geom if guidance != "none" else None,
            grad_clip=5.0)
        record("T1", "PCA-Proj", "Repulsion+Proj", gtype, pred)

    # ================================================================
    #  TIER 2: Cross-objective (fixed backbone = PVCNN)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  TIER 2: Cross-geometric-objective generalizability")
    print("=" * 70)

    for obj_name, obj_loss in geom_losses.items():
        print(f"\n[Objective] {obj_name}")
        for gtype in ["Standard", "Ortho"]:
            if gtype == "Standard":
                res = ddpm_denoise_guided(
                    trainer.model, trainer.sb_schedule, all_noisy,
                    sampling_steps=args.sampling_steps, guidance_scale=lam,
                    annealing="constant", geom_loss=obj_loss, grad_clip=5.0,
                    verbose=False)
            else:
                res = ddpm_denoise_ortho_guided(
                    trainer.model, trainer.sb_schedule, all_noisy,
                    sampling_steps=args.sampling_steps, guidance_scale=lam,
                    annealing="linear_decay", geom_loss=obj_loss, grad_clip=5.0,
                    verbose=False)
            record("T2", "PVCNN", obj_name, gtype, res["x_pred"])

    # ================================================================
    #  Summary tables
    # ================================================================

    # ── Tier 1 table ──
    print(f"\n\n{'=' * 100}")
    print(f"  TIER 1: Cross-Backbone — σ={args.noise_std}, λ={lam}, N={N}")
    print(f"{'=' * 100}")
    t1 = [r for r in all_results if r["tier"] == "T1"]
    headers = ["Backbone", "Guidance", "CD ↓", "VD ↓", "IGSD ↓"]
    rows = [[r["backbone"], r["guidance"],
             f"{r['CD']:.2f} ± {r['CD_std']:.2f}",
             f"{r['VD']:.4f} ± {r['VD_std']:.4f}",
             f"{r['IGSD']:.6f} ± {r['IGSD_std']:.6f}"]
            for r in t1]
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # ── Tier 1: Ortho improvement ──
    print(f"\n  Ortho improvement over Standard (per backbone):")
    for bb in ["PVCNN", "Bilateral", "PCA-Proj"]:
        std_r = next(r for r in t1 if r["backbone"] == bb and r["guidance"] == "Standard")
        ort_r = next(r for r in t1 if r["backbone"] == bb and r["guidance"] == "Ortho")
        dcd = (std_r["CD"] - ort_r["CD"]) / std_r["CD"] * 100
        dvd = (std_r["VD"] - ort_r["VD"]) / std_r["VD"] * 100
        dig = (std_r["IGSD"] - ort_r["IGSD"]) / std_r["IGSD"] * 100
        print(f"    {bb:<12s}  CD: {dcd:+.2f}%   VD: {dvd:+.2f}%   IGSD: {dig:+.2f}%")

    # ── Tier 2 table ──
    print(f"\n\n{'=' * 100}")
    print(f"  TIER 2: Cross-Objective — PVCNN backbone, σ={args.noise_std}, λ={lam}")
    print(f"{'=' * 100}")
    # Add baseline row
    base_r = next(r for r in all_results
                  if r["backbone"] == "PVCNN" and r["guidance"] == "Baseline")
    t2_rows = [["—", "Baseline",
                f"{base_r['CD']:.2f}", f"{base_r['VD']:.4f}",
                f"{base_r['IGSD']:.6f}"]]
    t2 = [r for r in all_results if r["tier"] == "T2"]
    for r in t2:
        t2_rows.append([r["objective"], r["guidance"],
                        f"{r['CD']:.2f} ± {r['CD_std']:.2f}",
                        f"{r['VD']:.4f} ± {r['VD_std']:.4f}",
                        f"{r['IGSD']:.6f} ± {r['IGSD_std']:.6f}"])
    headers2 = ["Objective", "Guidance", "CD ↓", "VD ↓", "IGSD ↓"]
    print(tabulate(t2_rows, headers=headers2, tablefmt="github"))

    # ── Tier 2: Ortho improvement ──
    print(f"\n  Ortho improvement over Standard (per objective):")
    for obj in geom_losses:
        std_r = next(r for r in t2 if r["objective"] == obj and r["guidance"] == "Standard")
        ort_r = next(r for r in t2 if r["objective"] == obj and r["guidance"] == "Ortho")
        dcd = (std_r["CD"] - ort_r["CD"]) / std_r["CD"] * 100
        dvd = (std_r["VD"] - ort_r["VD"]) / std_r["VD"] * 100
        dig = (std_r["IGSD"] - ort_r["IGSD"]) / std_r["IGSD"] * 100
        print(f"    {obj:<16s}  CD: {dcd:+.2f}%   VD: {dvd:+.2f}%   IGSD: {dig:+.2f}%")

    # Save
    out_file = os.path.join(args.out_dir, "generalizability_results.json")
    save = [{k: float(v) if isinstance(v, (np.floating,)) else v
             for k, v in r.items()} for r in all_results]
    with open(out_file, "w") as f:
        json.dump({"sigma": args.noise_std, "lambda": lam,
                   "num_shapes": N, "results": save}, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
