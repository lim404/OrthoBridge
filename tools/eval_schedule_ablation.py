"""Schedule ablation for orthogonal guidance.

Compares three annealing schedules on the SAME orthogonal guidance method:
  1. constant        — λ_t = λ_max  (no annealing)
  2. linear_rampup   — λ_t = λ_max · (1 - t/T)  (weak→strong as clean)
  3. snr_inverse     — λ_t = λ_max · SNR/(1+SNR)  (SNR-based ramp-up)

All use orthogonal projection + same geometric loss.
Tests λ_max ∈ {0.3, 3.0} at σ = 0.03.

Usage:
    python eval_schedule_ablation.py
    python eval_schedule_ablation.py --num_shapes 100 --lambdas 0.3 3.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate

from sb_cover.training.trainer_igv import TrainerIGV
from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
from sb_cover.evaluation.guided_sampling import (
    ddpm_denoise_ortho_guided, GeometricQualityLoss,
)
from sb_cover.data.punet_loader import get_punet_loaders
from models.train_utils import to_cuda
from metrics.geometric_metrics import compute_vd, compute_igsd


def compute_metrics(pred, gt, label=""):
    B = pred.shape[0]
    cds, vds, igsds = [], [], []
    for i in range(B):
        p, g = pred[i].T, gt[i].T  # (N, 3)
        d = torch.cdist(p.unsqueeze(0), g.unsqueeze(0)).squeeze(0)
        cds.append((d.min(1).values.mean() + d.min(0).values.mean()).item() * 1000)
        vds.append(compute_vd(p, g))
        igsds.append(compute_igsd(p, g))
        if (i + 1) % max(B // 5, 1) == 0:
            print(f"    [{label}] {i+1}/{B}  CD={cds[-1]:.2f}")
    return {k: np.array(v) for k, v in
            [("CD", cds), ("VD", vds), ("IGSD", igsds)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shapes", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.3, 3.0])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/PVDS_PUNet/latest.pth")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--out_dir", type=str,
                        default="experiments/schedule_ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone(args.checkpoint)
    trainer.model.eval()

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
    print(f"Test shapes: {N},  σ = {args.noise_std}\n")

    geom_loss = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0.0, subsample_n=512)

    schedules = [
        ("constant",       "constant"),
        ("linear_rampup",  "linear_rampup"),
        ("snr_inverse",    "snr_inverse"),
    ]

    all_results = []

    # Baseline
    print("=" * 70)
    print("[Baseline] No guidance")
    print("=" * 70)
    with torch.no_grad():
        res = ddpm_denoise(trainer.model, trainer.sb_schedule, all_noisy,
                           sampling_steps=args.sampling_steps, verbose=True)
    m = compute_metrics(res["x_pred"], all_clean, "Baseline")
    row = {"schedule": "Baseline", "lambda": 0.0}
    row.update({k: v.mean() for k, v in m.items()})
    row.update({k + "_std": v.std() for k, v in m.items()})
    all_results.append(row)
    print(f"  CD={row['CD']:.2f}  VD={row['VD']:.4f}  IGSD={row['IGSD']:.6f}\n")

    # Schedule × λ sweep
    for sched_name, sched_key in schedules:
        for lam in args.lambdas:
            print("=" * 70)
            print(f"[Ortho + {sched_name}] λ_max = {lam}")
            print("=" * 70)
            t0 = time.time()
            res = ddpm_denoise_ortho_guided(
                trainer.model, trainer.sb_schedule, all_noisy,
                sampling_steps=args.sampling_steps,
                guidance_scale=lam,
                annealing=sched_key,
                geom_loss=geom_loss,
                grad_clip=5.0,
                verbose=True,
            )
            dt = time.time() - t0
            m = compute_metrics(res["x_pred"], all_clean,
                                f"{sched_name}-{lam}")
            row = {"schedule": sched_name, "lambda": lam, "time_s": dt}
            row.update({k: v.mean() for k, v in m.items()})
            row.update({k + "_std": v.std() for k, v in m.items()})
            all_results.append(row)
            print(f"  CD={row['CD']:.2f}  VD={row['VD']:.4f}"
                  f"  IGSD={row['IGSD']:.6f}  ({dt:.1f}s)\n")

    # ── Summary ──
    print("\n" + "=" * 95)
    print(f"  SCHEDULE ABLATION — Orthogonal Guidance, σ = {args.noise_std}, "
          f"N = {N}")
    print("=" * 95)

    headers = ["Schedule", "λ_max", "CD ↓ (×1e3)", "VD ↓", "IGSD ↓"]
    rows = []
    for r in all_results:
        rows.append([
            r["schedule"],
            f"{r['lambda']:.1f}" if r["lambda"] > 0 else "—",
            f"{r['CD']:.2f} ± {r['CD_std']:.2f}",
            f"{r['VD']:.4f} ± {r['VD_std']:.4f}",
            f"{r['IGSD']:.6f} ± {r['IGSD_std']:.6f}",
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # Δ vs baseline
    base = all_results[0]
    print(f"\n  Δ vs Baseline:")
    for r in all_results[1:]:
        dcd = (base["CD"] - r["CD"]) / base["CD"] * 100
        dvd = (base["VD"] - r["VD"]) / base["VD"] * 100
        digsd = (base["IGSD"] - r["IGSD"]) / base["IGSD"] * 100
        print(f"    {r['schedule']:>16s} λ={r['lambda']:<4.1f}"
              f"  CD: {dcd:+6.2f}%   VD: {dvd:+6.2f}%   IGSD: {digsd:+6.2f}%")

    # Save
    out_file = os.path.join(args.out_dir,
                            f"schedule_ablation_sigma{args.noise_std}_n{N}.json")
    with open(out_file, "w") as f:
        json.dump({"sigma": args.noise_std, "num_shapes": N,
                   "results": [{k: float(v) if isinstance(v, (np.floating,))
                                else v for k, v in r.items()}
                               for r in all_results]}, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
