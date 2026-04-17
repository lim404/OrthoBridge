"""External calibration of VD and IGSD metrics.

Validates that our novel metrics correlate with established geometric
quality measures:
  1. IGSD vs P2M  (silhouette consistency ↔ surface proximity)
  2. VD   vs kNN-var  (volumetric fidelity ↔ local uniformity)

Generates denoised outputs at multiple quality levels (varying guidance
method and λ), then computes all metrics and Spearman/Pearson correlations.

Usage:
    python eval_metric_calibration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
import torch
from omegaconf import OmegaConf
from scipy import stats

from sb_cover.training.trainer_igv import TrainerIGV
from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
from sb_cover.evaluation.guided_sampling import (
    ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss,
)
from sb_cover.data.punet_loader import get_punet_loaders
from models.train_utils import to_cuda
from metrics.geometric_metrics import compute_vd, compute_igsd


# ──────────────────────────────────────────────────────────────────── #
#  Extra metric computation
# ──────────────────────────────────────────────────────────────────── #

def compute_p2m(pred_pts: torch.Tensor, verts: torch.Tensor,
                faces: torch.Tensor) -> float:
    """Point-to-mesh distance (mean nearest-face-center distance).

    pred_pts: (N, 3), verts: (V, 3), faces: (F, 3).
    Returns mean distance (lower = better).
    """
    # Use face centers as proxy (fast, avoids pytorch3d dependency issues)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_centers = (v0 + v1 + v2) / 3.0  # (F, 3)

    # Nearest face center distance per point
    dist = torch.cdist(pred_pts.unsqueeze(0),
                       face_centers.unsqueeze(0)).squeeze(0)  # (N, F)
    min_dist = dist.min(dim=1).values  # (N,)
    return min_dist.mean().item()


def compute_knn_variance(pts: torch.Tensor, k: int = 6) -> float:
    """kNN distance variance — measures local uniformity.

    Lower variance = more uniform distribution.
    pts: (N, 3).
    """
    dist = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
    knn_dist, _ = dist.topk(k + 1, dim=-1, largest=False)
    knn_dist = knn_dist[:, 1:]  # exclude self
    mean_knn = knn_dist.mean(dim=1)  # per-point mean kNN distance
    return mean_knn.var().item()


def compute_cd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Chamfer Distance. pred, gt: (N, 3). Returns ×1000."""
    d = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    return (d.min(1).values.mean() + d.min(0).values.mean()).item() * 1000


# ──────────────────────────────────────────────────────────────────── #
#  Main
# ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_objects", type=int, default=10)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/PVDS_PUNet/latest.pth")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--out_dir", type=str,
                        default="figures/metric_calibration")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone(args.checkpoint)
    trainer.model.eval()
    print("Model loaded.\n")

    mesh_dir = os.path.join(cfg.data.data_dir, "PUNet", "meshes", "test")
    pcl_dir = os.path.join(cfg.data.data_dir, "PUNet", "pointclouds", "test",
                           "50000_poisson")

    mesh_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith(".off")])
    obj_names = [f[:-4] for f in mesh_files][:args.num_objects]

    geom_loss = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0.0, subsample_n=512)

    # Quality levels: different methods × λ → diverse quality spectrum
    quality_configs = [
        ("Noisy input",        None,       0.0),
        ("Baseline",           "none",     0.0),
        ("Standard λ=1",      "standard",  1.0),
        ("Standard λ=5",      "standard",  5.0),
        ("Standard λ=10",     "standard",  10.0),
        ("Standard λ=20",     "standard",  20.0),
        ("Ortho λ=1",         "ortho",     1.0),
        ("Ortho λ=5",         "ortho",     5.0),
        ("Ortho λ=10",        "ortho",     10.0),
        ("Ortho λ=20",        "ortho",     20.0),
    ]

    # Accumulate per-sample metrics across all objects and quality levels
    all_samples = []  # list of dicts

    for oi, name in enumerate(obj_names):
        print(f"\n[{oi+1}/{len(obj_names)}] {name}")

        # Load mesh
        mesh_path = os.path.join(mesh_dir, name + ".off")
        v_raw, f_raw = pcu.load_mesh_vf(mesh_path)
        v_raw = torch.tensor(v_raw, dtype=torch.float32)
        f_raw = torch.tensor(f_raw, dtype=torch.long)

        # Load clean point cloud
        pcl = torch.tensor(np.loadtxt(
            os.path.join(pcl_dir, name + ".xyz"), dtype=np.float32))

        # Normalize mesh and pcl to unit sphere (same transform)
        p_max = pcl.max(0, keepdim=True).values
        p_min = pcl.min(0, keepdim=True).values
        center = (p_max + p_min) / 2
        pcl_c = pcl - center
        scale = (pcl_c ** 2).sum(1, keepdim=True).sqrt().max().item()
        pcl_c = pcl_c / scale
        v_norm = (v_raw - center) / scale

        # Subsample to npoints
        idx = torch.randperm(pcl_c.shape[0])[:args.npoints]
        gt_pts = pcl_c[idx]  # (N, 3)
        noisy_pts = gt_pts + torch.randn_like(gt_pts) * args.noise_std

        x_noisy = noisy_pts.T.unsqueeze(0).to(device)  # (1, 3, N)
        x_clean = gt_pts.T.unsqueeze(0).to(device)

        for q_name, method, lam in quality_configs:
            if method is None:
                # Use noisy input directly
                pred_b3n = x_noisy
            elif method == "none":
                with torch.no_grad():
                    res = ddpm_denoise(trainer.model, trainer.sb_schedule,
                                       x_noisy, sampling_steps=args.sampling_steps,
                                       verbose=False)
                pred_b3n = res["x_pred"]
            elif method == "standard":
                res = ddpm_denoise_guided(
                    trainer.model, trainer.sb_schedule, x_noisy,
                    sampling_steps=args.sampling_steps,
                    guidance_scale=lam, annealing="constant",
                    geom_loss=geom_loss, grad_clip=5.0, verbose=False)
                pred_b3n = res["x_pred"]
            elif method == "ortho":
                res = ddpm_denoise_ortho_guided(
                    trainer.model, trainer.sb_schedule, x_noisy,
                    sampling_steps=args.sampling_steps,
                    guidance_scale=lam, annealing="linear_decay",
                    geom_loss=geom_loss, grad_clip=5.0, verbose=False)
                pred_b3n = res["x_pred"]

            pred_pts = pred_b3n[0].T.cpu()  # (N, 3)
            gt_cpu = x_clean[0].T.cpu()

            sample = {
                "object": name,
                "method": q_name,
                "CD": compute_cd(pred_pts, gt_cpu),
                "VD": compute_vd(pred_pts, gt_cpu),
                "IGSD": compute_igsd(pred_pts, gt_cpu),
                "P2M": compute_p2m(pred_pts, v_norm, f_raw),
                "kNN_var": compute_knn_variance(pred_pts),
            }
            all_samples.append(sample)

        print(f"  {len(quality_configs)} quality levels computed")

    # ── Correlation analysis ──
    print(f"\n{'=' * 70}")
    print(f"METRIC CALIBRATION — {len(all_samples)} data points "
          f"({len(obj_names)} objects × {len(quality_configs)} quality levels)")
    print(f"{'=' * 70}\n")

    metrics = {k: np.array([s[k] for s in all_samples])
               for k in ["CD", "VD", "IGSD", "P2M", "kNN_var"]}

    # Correlation pairs
    pairs = [
        ("IGSD", "P2M",     "IGSD vs P2M (silhouette ↔ surface proximity)"),
        ("IGSD", "CD",      "IGSD vs CD"),
        ("VD",   "kNN_var", "VD vs kNN-var (volumetric ↔ uniformity)"),
        ("VD",   "CD",      "VD vs CD"),
        ("VD",   "P2M",     "VD vs P2M"),
        ("IGSD", "kNN_var", "IGSD vs kNN-var"),
    ]

    print(f"{'Pair':<45} {'Spearman ρ':>12} {'p-value':>10} {'Pearson r':>12}")
    print("-" * 85)
    corr_results = []
    for m1, m2, desc in pairs:
        sp = stats.spearmanr(metrics[m1], metrics[m2])
        pr = stats.pearsonr(metrics[m1], metrics[m2])
        sig = "***" if sp.pvalue < 0.001 else "**" if sp.pvalue < 0.01 \
              else "*" if sp.pvalue < 0.05 else ""
        print(f"{desc:<45} {sp.statistic:>+10.4f}{sig:>2s} "
              f"{sp.pvalue:>10.2e} {pr.statistic:>+10.4f}")
        corr_results.append({
            "pair": desc, "m1": m1, "m2": m2,
            "spearman_rho": sp.statistic, "spearman_p": sp.pvalue,
            "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
        })

    # ── Scatter plots ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for ax, (m1, m2, desc) in zip(axes.flat, pairs):
        x, y = metrics[m1], metrics[m2]
        sp = stats.spearmanr(x, y)

        # Color by method type
        method_names = [s["method"] for s in all_samples]
        colors_map = {
            "Noisy input": "#9E9E9E",
            "Baseline": "#4CAF50",
        }
        for mn in quality_configs:
            if "Standard" in mn[0]:
                colors_map[mn[0]] = "#F44336"
            elif "Ortho" in mn[0]:
                colors_map[mn[0]] = "#2196F3"

        colors = [colors_map.get(mn, "#999") for mn in method_names]
        ax.scatter(x, y, c=colors, alpha=0.6, s=25, edgecolors="white",
                   linewidths=0.3)

        # Trend line
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), "--", color="black", lw=1, alpha=0.5)

        ax.set_xlabel(m1, fontsize=11)
        ax.set_ylabel(m2, fontsize=11)
        ax.set_title(f"{desc}\nSpearman ρ = {sp.statistic:.3f} "
                     f"(p = {sp.pvalue:.1e})",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#9E9E9E", label="Noisy input"),
        Patch(facecolor="#4CAF50", label="Baseline (no guidance)"),
        Patch(facecolor="#F44336", label="Standard guidance"),
        Patch(facecolor="#2196F3", label="Orthogonal guidance"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Metric External Calibration: Novel Metrics vs Established Measures",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_pdf = os.path.join(args.out_dir, "metric_calibration.pdf")
    out_png = out_pdf.replace(".pdf", ".png")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved: {out_pdf}")

    # Save data
    with open(os.path.join(args.out_dir, "metric_calibration_data.json"), "w") as f:
        json.dump({
            "samples": [{k: float(v) if isinstance(v, (float, np.floating))
                         else v for k, v in s.items()} for s in all_samples],
            "correlations": corr_results,
        }, f, indent=2)
    print(f"Data saved to {args.out_dir}/metric_calibration_data.json")


if __name__ == "__main__":
    main()
