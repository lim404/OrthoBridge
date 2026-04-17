"""Same-backbone guidance mechanism ablation.

Compares four guidance strategies on the SAME frozen P2P-Bridge backbone:
  1. Baseline       — vanilla DDPM reverse (no guidance)
  2. Standard       — full gradient guidance, constant annealing
  3. Clipped/Norm   — gradient clipped + cosine-annealed guidance
  4. Orthogonal     — gradient projected ⊥ to score direction

All use IDENTICAL geometric loss and backbone.  Only the gradient
post-processing and annealing differ → isolates the contribution of
"how guidance is injected."

Usage:
    python eval_guidance_ablation.py --noise_std 0.03
    python eval_guidance_ablation.py --noise_std 0.03 0.05 --lambdas 1 5 10 20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate

from sb_cover.training.trainer_igv import TrainerIGV
from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
from sb_cover.evaluation.guided_sampling import (
    ddpm_denoise_guided,
    ddpm_denoise_ortho_guided,
    GeometricQualityLoss,
    compute_lambda,
)
from sb_cover.data.punet_loader import get_punet_loaders
from models.train_utils import to_cuda
from metrics.geometric_metrics import compute_vd, compute_igsd


# ──────────────────────────────────────────────────────────────────── #
#  Per-sample metric computation
# ──────────────────────────────────────────────────────────────────── #

def compute_metrics_batch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    label: str = "",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute CD, VD, IGSD per sample.  pred, gt: (B, 3, N)."""
    B = pred.shape[0]
    cds, vds, igsds = [], [], []

    for i in range(B):
        p = pred[i].transpose(0, 1)  # (N, 3)
        g = gt[i].transpose(0, 1)

        # CD (×1000 for readability)
        d = torch.cdist(p.unsqueeze(0), g.unsqueeze(0)).squeeze(0)
        cd = (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item() * 1000
        cds.append(cd)

        # VD & IGSD
        vds.append(compute_vd(p, g))
        igsds.append(compute_igsd(p, g))

        if verbose and (i + 1) % max(B // 5, 1) == 0:
            print(f"    [{label}] {i+1}/{B}  "
                  f"CD={cd:.2f}  VD={vds[-1]:.4f}  IGSD={igsds[-1]:.6f}")

    return {
        "CD": np.array(cds),
        "VD": np.array(vds),
        "IGSD": np.array(igsds),
    }


# ──────────────────────────────────────────────────────────────────── #
#  Guidance runners
# ──────────────────────────────────────────────────────────────────── #

def run_baseline(model, sb_schedule, x_noisy, steps):
    with torch.no_grad():
        res = ddpm_denoise(model, sb_schedule, x_noisy,
                           sampling_steps=steps, verbose=True)
    return res["x_pred"]


def run_standard_guidance(model, sb_schedule, x_noisy, steps, lam, geom_loss):
    """Full gradient, constant annealing (no decay → exposes instability)."""
    res = ddpm_denoise_guided(
        model, sb_schedule, x_noisy,
        sampling_steps=steps,
        guidance_scale=lam,
        annealing="constant",
        geom_loss=geom_loss,
        grad_clip=5.0,
        verbose=True,
    )
    return res["x_pred"]


def run_clipped_guidance(model, sb_schedule, x_noisy, steps, lam, geom_loss):
    """Clipped gradient + cosine-annealed (conservative but stable)."""
    res = ddpm_denoise_guided(
        model, sb_schedule, x_noisy,
        sampling_steps=steps,
        guidance_scale=lam,
        annealing="cosine_decay",
        geom_loss=geom_loss,
        grad_clip=0.5,
        verbose=True,
    )
    return res["x_pred"]


def run_ortho_guidance(model, sb_schedule, x_noisy, steps, lam, geom_loss):
    """Orthogonal projection, linear decay (our method)."""
    res = ddpm_denoise_ortho_guided(
        model, sb_schedule, x_noisy,
        sampling_steps=steps,
        guidance_scale=lam,
        annealing="linear_decay",
        geom_loss=geom_loss,
        grad_clip=5.0,
        verbose=True,
    )
    return res["x_pred"]


# ──────────────────────────────────────────────────────────────────── #
#  Main
# ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Same-backbone guidance ablation")
    parser.add_argument("--num_shapes", type=int, default=100)
    parser.add_argument("--noise_std", type=float, nargs="+",
                        default=[0.03],
                        help="Noise σ levels to sweep")
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[1.0, 5.0, 10.0, 20.0],
                        help="Guidance scales to sweep")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/PVDS_PUNet/latest.pth")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--out_dir", type=str,
                        default="experiments/guidance_ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone(args.checkpoint)
    trainer.model.eval()
    print(f"Model loaded from {args.checkpoint}\n")

    geom_loss = GeometricQualityLoss(
        w_repulsion=1.0,
        w_projection=1.0,
        w_covariance=0.0,
        subsample_n=512,
    )

    guidance_methods = [
        ("Standard",     run_standard_guidance),
        ("Clipped/Norm", run_clipped_guidance),
        ("Orthogonal",   run_ortho_guidance),
    ]

    # ── Sweep over noise levels ──
    for sigma in args.noise_std:
        print(f"\n{'#' * 80}")
        print(f"  NOISE LEVEL σ = {sigma}")
        print(f"{'#' * 80}\n")

        # Load test data with fixed σ
        _, test_loader = get_punet_loaders(
            data_dir=cfg.data.data_dir,
            patch_size=cfg.data.get("npoints", 2048),
            batch_size=args.batch_size,
            noise_min=sigma,
            noise_max=sigma,
            num_workers=2,
            num_patches=10,
        )

        all_clean, all_noisy = [], []
        n = 0
        for batch in test_loader:
            if n >= args.num_shapes:
                break
            batch = to_cuda(batch, device)
            x_clean, x_noisy = trainer._extract_batch(batch)
            use = min(x_clean.shape[0], args.num_shapes - n)
            all_clean.append(x_clean[:use])
            all_noisy.append(x_noisy[:use])
            n += use
        all_clean = torch.cat(all_clean, 0)
        all_noisy = torch.cat(all_noisy, 0)
        N = all_clean.shape[0]
        print(f"Test shapes: {N},  σ = {sigma}\n")

        all_results = []

        # 1. Baseline
        print("=" * 70)
        print("[Baseline] P2P-Bridge (no guidance)")
        print("=" * 70)
        t0 = time.time()
        pred_base = run_baseline(
            trainer.model, trainer.sb_schedule, all_noisy, args.sampling_steps)
        dt = time.time() - t0
        m = compute_metrics_batch(pred_base, all_clean, "Baseline")
        row = {"method": "Baseline", "lambda": 0.0, "time_s": dt}
        row.update({k: v.mean() for k, v in m.items()})
        row.update({k + "_std": v.std() for k, v in m.items()})
        all_results.append(row)
        print(f"  CD={row['CD']:.2f}  VD={row['VD']:.4f}  "
              f"IGSD={row['IGSD']:.6f}  ({dt:.1f}s)\n")

        # 2-4. Guided methods × λ sweep
        for method_name, method_fn in guidance_methods:
            for lam in args.lambdas:
                print("=" * 70)
                print(f"[{method_name}] λ = {lam}")
                print("=" * 70)
                t0 = time.time()
                pred = method_fn(
                    trainer.model, trainer.sb_schedule, all_noisy,
                    args.sampling_steps, lam, geom_loss,
                )
                dt = time.time() - t0
                m = compute_metrics_batch(
                    pred, all_clean, f"{method_name}-{lam}")
                row = {"method": method_name, "lambda": lam, "time_s": dt}
                row.update({k: v.mean() for k, v in m.items()})
                row.update({k + "_std": v.std() for k, v in m.items()})
                all_results.append(row)
                print(f"  CD={row['CD']:.2f}  VD={row['VD']:.4f}  "
                      f"IGSD={row['IGSD']:.6f}  ({dt:.1f}s)\n")

        # ── Summary table ──
        print("\n" + "=" * 100)
        print(f"  GUIDANCE ABLATION — σ = {sigma},  N = {N} shapes,  "
              f"steps = {args.sampling_steps}")
        print("=" * 100)

        headers = ["Method", "λ", "CD ↓ (×1e3)", "VD ↓", "IGSD ↓"]
        table_rows = []
        for r in all_results:
            table_rows.append([
                r["method"],
                f"{r['lambda']:.1f}" if r["lambda"] > 0 else "—",
                f"{r['CD']:.2f} ± {r['CD_std']:.2f}",
                f"{r['VD']:.4f} ± {r['VD_std']:.4f}",
                f"{r['IGSD']:.6f} ± {r['IGSD_std']:.6f}",
            ])
        print(tabulate(table_rows, headers=headers, tablefmt="github"))

        # ── Deltas vs baseline ──
        base = all_results[0]
        print(f"\n  Δ vs Baseline (positive = improvement):")
        for r in all_results[1:]:
            dcd  = (base["CD"]   - r["CD"])   / base["CD"]   * 100
            dvd  = (base["VD"]   - r["VD"])   / base["VD"]   * 100
            digsd = (base["IGSD"] - r["IGSD"]) / base["IGSD"] * 100
            tag = f"{r['method']:>14s} λ={r['lambda']:<4.1f}"
            print(f"    {tag}  CD: {dcd:+6.2f}%   VD: {dvd:+6.2f}%   "
                  f"IGSD: {digsd:+6.2f}%")

        # ── Best per metric ──
        print()
        for mn in ["CD", "VD", "IGSD"]:
            vals = [(r["method"], r["lambda"], r[mn]) for r in all_results]
            best = min(vals, key=lambda x: x[2])
            print(f"  Best {mn}: {best[0]} (λ={best[1]}) → {best[2]:.6f}")
        print()

        # ── Save JSON ──
        out_file = os.path.join(
            args.out_dir,
            f"guidance_ablation_sigma{sigma}_n{N}.json",
        )
        serialisable = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                sr[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            serialisable.append(sr)
        with open(out_file, "w") as f:
            json.dump({"sigma": sigma, "num_shapes": N,
                       "sampling_steps": args.sampling_steps,
                       "results": serialisable}, f, indent=2)
        print(f"  Results saved to {out_file}")


if __name__ == "__main__":
    main()
