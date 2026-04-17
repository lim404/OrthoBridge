"""Head-to-head: P2P-Bridge baseline  vs  Orth-0.2 on VD / IGSD / CD.

Usage:
    python eval_p2pb_vs_orth.py --num_shapes 50
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

from sb_cover.training.trainer_igv import TrainerIGV
from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
from sb_cover.evaluation.guided_sampling import (
    ddpm_denoise_ortho_guided, GeometricQualityLoss)
from sb_cover.data.punet_loader import get_punet_loaders
from models.train_utils import to_cuda
from metrics.geometric_metrics import compute_vd, compute_igsd


def evaluate(label, x_pred, x_clean):
    """Per-sample VD / IGSD / CD → arrays."""
    B = x_pred.shape[0]
    vds, igsds, cds = [], [], []
    for i in range(B):
        pred_i  = x_pred[i].transpose(0, 1)   # (N, 3)
        clean_i = x_clean[i].transpose(0, 1)
        vds.append(compute_vd(pred_i, clean_i))
        igsds.append(compute_igsd(pred_i, clean_i))
        cds.append(((x_pred[i] - x_clean[i]) ** 2).sum(0).mean().item() * 1000)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{B}] VD={vds[-1]:.4f}  IGSD={igsds[-1]:.6f}")
    return np.array(vds), np.array(igsds), np.array(cds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shapes", type=int, default=50)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=0.2)
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/shapenet_denoise_sb_igv.yaml")
    device = torch.device("cuda")

    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    _, test_loader = get_punet_loaders(
        data_dir=cfg.data.data_dir,
        patch_size=cfg.data.get("npoints", 2048),
        batch_size=8,
        noise_min=cfg.data.get("noise_min", 0.01),
        noise_max=cfg.data.get("noise_max", 0.02),
        num_workers=2,
        num_patches=10,
    )

    # Collect test data into one big batch
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
    print(f"Test shapes: {N}\n")

    geom_loss = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    # ── 1. P2P-Bridge baseline ──
    print("=" * 60)
    print("[1/2] P2P-Bridge (baseline DDPM)")
    print("=" * 60)
    with torch.no_grad():
        res_base = ddpm_denoise(
            trainer.model, trainer.sb_schedule, all_noisy,
            sampling_steps=args.sampling_steps, verbose=True)
    vd_b, igsd_b, cd_b = evaluate("P2P-Bridge", res_base["x_pred"], all_clean)

    # ── 2. Orth-0.2 ──
    print("\n" + "=" * 60)
    print(f"[2/2] Orth-{args.guidance_scale}")
    print("=" * 60)
    res_orth = ddpm_denoise_ortho_guided(
        trainer.model, trainer.sb_schedule, all_noisy,
        sampling_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        annealing="linear_decay",
        geom_loss=geom_loss,
        grad_clip=2.0,
        verbose=True)
    vd_o, igsd_o, cd_o = evaluate("Orth", res_orth["x_pred"], all_clean)

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'VD ↓':>12} {'IGSD ↓':>14} {'CD ↓':>12}")
    print("-" * 70)
    print(f"{'P2P-Bridge':<20} {vd_b.mean():>8.4f} ± {vd_b.std():<5.4f}"
          f"{igsd_b.mean():>10.6f} ± {igsd_b.std():<8.6f}"
          f"{cd_b.mean():>8.1f} ± {cd_b.std():<6.1f}")
    print(f"{'Orth-' + str(args.guidance_scale):<20} {vd_o.mean():>8.4f} ± {vd_o.std():<5.4f}"
          f"{igsd_o.mean():>10.6f} ± {igsd_o.std():<8.6f}"
          f"{cd_o.mean():>8.1f} ± {cd_o.std():<6.1f}")
    print("-" * 70)

    # Improvement
    dvd  = (vd_b.mean() - vd_o.mean()) / vd_b.mean() * 100
    digsd = (igsd_b.mean() - igsd_o.mean()) / igsd_b.mean() * 100
    dcd  = (cd_b.mean() - cd_o.mean()) / cd_b.mean() * 100
    winner_vd   = "Orth" if dvd > 0 else "P2P-Bridge"
    winner_igsd = "Orth" if digsd > 0 else "P2P-Bridge"
    winner_cd   = "Orth" if dcd > 0 else "P2P-Bridge"
    print(f"\n  VD   improvement: {dvd:+.1f}%  →  {winner_vd} wins")
    print(f"  IGSD improvement: {digsd:+.1f}%  →  {winner_igsd} wins")
    print(f"  CD   improvement: {dcd:+.1f}%  →  {winner_cd} wins")
    print("=" * 70)


if __name__ == "__main__":
    main()
