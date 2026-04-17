"""Multi-seed stability evaluation.

Runs key experiments across 5 random seeds to report mean ± std,
addressing reviewer concerns about seed sensitivity.

Covers:
  - Baseline (no guidance)
  - Standard guidance λ=3.0 (constant annealing — stress test)
  - Orthogonal guidance λ=3.0 (linear_decay — ours)

At σ = 0.03, 100 shapes per seed.

Usage:
    python eval_multiseed.py
    python eval_multiseed.py --seeds 42 123 456 789 1024 --num_shapes 100
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate

from sb_cover.training.trainer_igv import TrainerIGV
from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
from sb_cover.evaluation.guided_sampling import (
    ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss,
)
from sb_cover.data.punet_loader import get_punet_loaders
from models.train_utils import to_cuda
from metrics.geometric_metrics import compute_vd, compute_igsd


def compute_metrics(pred, gt):
    """Returns per-sample CD, VD, IGSD arrays."""
    B = pred.shape[0]
    cds, vds, igsds = [], [], []
    for i in range(B):
        p, g = pred[i].T, gt[i].T
        d = torch.cdist(p.unsqueeze(0), g.unsqueeze(0)).squeeze(0)
        cds.append((d.min(1).values.mean() + d.min(0).values.mean()).item() * 1e3)
        vds.append(compute_vd(p, g))
        igsds.append(compute_igsd(p, g))
    return np.array(cds), np.array(vds), np.array(igsds)


def run_one_seed(trainer, cfg, device, seed, args):
    """Run all three methods with a fixed seed. Returns dict of results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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

    geom_loss = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0.0, subsample_n=512)

    results = {}

    # 1. Baseline
    with torch.no_grad():
        res = ddpm_denoise(trainer.model, trainer.sb_schedule, all_noisy,
                           sampling_steps=args.sampling_steps, verbose=False)
    cd, vd, igsd = compute_metrics(res["x_pred"], all_clean)
    results["Baseline"] = {"CD": cd, "VD": vd, "IGSD": igsd}

    # 2. Standard guidance λ=3.0 (constant — stress test)
    res = ddpm_denoise_guided(
        trainer.model, trainer.sb_schedule, all_noisy,
        sampling_steps=args.sampling_steps,
        guidance_scale=3.0, annealing="constant",
        geom_loss=geom_loss, grad_clip=5.0, verbose=False)
    cd, vd, igsd = compute_metrics(res["x_pred"], all_clean)
    results["Standard λ=3"] = {"CD": cd, "VD": vd, "IGSD": igsd}

    # 3. Orthogonal guidance λ=3.0 (linear_decay — ours)
    res = ddpm_denoise_ortho_guided(
        trainer.model, trainer.sb_schedule, all_noisy,
        sampling_steps=args.sampling_steps,
        guidance_scale=3.0, annealing="linear_decay",
        geom_loss=geom_loss, grad_clip=5.0, verbose=False)
    cd, vd, igsd = compute_metrics(res["x_pred"], all_clean)
    results["Ortho λ=3 (ours)"] = {"CD": cd, "VD": vd, "IGSD": igsd}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--num_shapes", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/PVDS_PUNet/latest.pth")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--out_dir", type=str,
                        default="experiments/multiseed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone(args.checkpoint)
    trainer.model.eval()
    print(f"Model loaded. Seeds: {args.seeds}\n")

    # method → metric → list of per-seed means
    agg = defaultdict(lambda: defaultdict(list))
    # method → metric → list of per-seed per-sample arrays (for pooled std)
    all_samples = defaultdict(lambda: defaultdict(list))

    for si, seed in enumerate(args.seeds):
        print(f"{'=' * 60}")
        print(f"  Seed {seed}  [{si+1}/{len(args.seeds)}]")
        print(f"{'=' * 60}")
        t0 = time.time()
        results = run_one_seed(trainer, cfg, device, seed, args)
        dt = time.time() - t0

        for method, metrics in results.items():
            for mk in ["CD", "VD", "IGSD"]:
                agg[method][mk].append(metrics[mk].mean())
                all_samples[method][mk].append(metrics[mk])
            print(f"  {method:<20s}  CD={metrics['CD'].mean():.2f}  "
                  f"VD={metrics['VD'].mean():.4f}  "
                  f"IGSD={metrics['IGSD'].mean():.6f}")
        print(f"  ({dt:.1f}s)\n")

    # ── Summary ──
    methods = ["Baseline", "Standard λ=3", "Ortho λ=3 (ours)"]
    n_seeds = len(args.seeds)

    print("\n" + "=" * 100)
    print(f"  MULTI-SEED STABILITY — σ = {args.noise_std}, "
          f"N = {args.num_shapes} shapes, {n_seeds} seeds")
    print("=" * 100)

    # Table 1: Cross-seed mean ± std of per-seed means
    headers = ["Method", "CD ↓ (×1e3)", "VD ↓", "IGSD ↓"]
    rows = []
    for m in methods:
        cd_arr = np.array(agg[m]["CD"])
        vd_arr = np.array(agg[m]["VD"])
        ig_arr = np.array(agg[m]["IGSD"])
        rows.append([
            m,
            f"{cd_arr.mean():.2f} ± {cd_arr.std():.2f}",
            f"{vd_arr.mean():.4f} ± {vd_arr.std():.4f}",
            f"{ig_arr.mean():.6f} ± {ig_arr.std():.6f}",
        ])
    print("\nCross-seed aggregation (mean ± std of seed means):\n")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # Table 2: Pooled statistics (all samples across seeds)
    print("\nPooled across all seeds (total samples = "
          f"{args.num_shapes} × {n_seeds} = {args.num_shapes * n_seeds}):\n")
    rows2 = []
    for m in methods:
        cd_pool = np.concatenate(all_samples[m]["CD"])
        vd_pool = np.concatenate(all_samples[m]["VD"])
        ig_pool = np.concatenate(all_samples[m]["IGSD"])
        rows2.append([
            m,
            f"{cd_pool.mean():.2f} ± {cd_pool.std():.2f}",
            f"{vd_pool.mean():.4f} ± {vd_pool.std():.4f}",
            f"{ig_pool.mean():.6f} ± {ig_pool.std():.6f}",
        ])
    print(tabulate(rows2, headers=headers, tablefmt="github"))

    # Coefficient of variation across seeds
    print("\nCoefficient of Variation (CV = σ/μ, lower = more stable):\n")
    for m in methods:
        cvs = {}
        for mk in ["CD", "VD", "IGSD"]:
            arr = np.array(agg[m][mk])
            cvs[mk] = arr.std() / arr.mean() * 100 if arr.mean() > 0 else 0
        print(f"  {m:<20s}  CD: {cvs['CD']:.2f}%   "
              f"VD: {cvs['VD']:.2f}%   IGSD: {cvs['IGSD']:.2f}%")

    # Improvement stats
    print(f"\nΔ vs Baseline (pooled, positive = improvement):")
    base_cd = np.concatenate(all_samples["Baseline"]["CD"])
    base_vd = np.concatenate(all_samples["Baseline"]["VD"])
    base_ig = np.concatenate(all_samples["Baseline"]["IGSD"])
    for m in methods[1:]:
        cd_p = np.concatenate(all_samples[m]["CD"])
        vd_p = np.concatenate(all_samples[m]["VD"])
        ig_p = np.concatenate(all_samples[m]["IGSD"])
        dcd = (base_cd.mean() - cd_p.mean()) / base_cd.mean() * 100
        dvd = (base_vd.mean() - vd_p.mean()) / base_vd.mean() * 100
        dig = (base_ig.mean() - ig_p.mean()) / base_ig.mean() * 100
        print(f"  {m:<20s}  CD: {dcd:+.2f}%   VD: {dvd:+.2f}%   "
              f"IGSD: {dig:+.2f}%")

    # Save
    save_data = {
        "config": {"sigma": args.noise_std, "num_shapes": args.num_shapes,
                    "seeds": args.seeds, "sampling_steps": args.sampling_steps},
        "per_seed": {},
        "cross_seed": {},
        "pooled": {},
    }
    for m in methods:
        save_data["per_seed"][m] = {
            mk: [float(v) for v in agg[m][mk]] for mk in ["CD", "VD", "IGSD"]}
        save_data["cross_seed"][m] = {
            mk: {"mean": float(np.mean(agg[m][mk])),
                 "std": float(np.std(agg[m][mk]))}
            for mk in ["CD", "VD", "IGSD"]}
        save_data["pooled"][m] = {
            mk: {"mean": float(np.concatenate(all_samples[m][mk]).mean()),
                 "std": float(np.concatenate(all_samples[m][mk]).std())}
            for mk in ["CD", "VD", "IGSD"]}

    out_file = os.path.join(args.out_dir, "multiseed_results.json")
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
