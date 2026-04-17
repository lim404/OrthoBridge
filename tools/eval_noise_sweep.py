"""P2P-Bridge vs Orth-2.0: VD / IGSD across noise levels σ = 0.01, 0.02, 0.03.

Usage:
    python eval_noise_sweep.py --num_shapes 50
    python eval_noise_sweep.py --num_shapes 50 --resolution 50000_poisson
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


def evaluate(x_pred, x_clean):
    B = x_pred.shape[0]
    vds, igsds = [], []
    for i in range(B):
        pred_i  = x_pred[i].transpose(0, 1)
        clean_i = x_clean[i].transpose(0, 1)
        vds.append(compute_vd(pred_i, clean_i))
        igsds.append(compute_igsd(pred_i, clean_i))
    return np.array(vds), np.array(igsds)


def collect_data(trainer, cfg, sigma, num_shapes, device, resolution=None):
    """Load test patches with fixed noise level sigma."""
    resolutions = [resolution] if resolution else None
    _, test_loader = get_punet_loaders(
        data_dir=cfg.data.data_dir,
        patch_size=cfg.data.get("npoints", 2048),
        batch_size=8,
        noise_min=sigma, noise_max=sigma,   # fixed noise
        num_workers=2,
        num_patches=10,
        resolutions=resolutions,
    )
    all_clean, all_noisy = [], []
    n = 0
    for batch in test_loader:
        if n >= num_shapes:
            break
        batch = to_cuda(batch, device)
        x_clean, x_noisy = trainer._extract_batch(batch)
        use = min(x_clean.shape[0], num_shapes - n)
        all_clean.append(x_clean[:use])
        all_noisy.append(x_noisy[:use])
        n += use
    return torch.cat(all_clean, 0), torch.cat(all_noisy, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shapes", type=int, default=50)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--resolution", type=str, default=None,
                        help="e.g. 50000_poisson (default: all resolutions)")
    args = parser.parse_args()

    cfg = OmegaConf.load("configs/shapenet_denoise_sb_igv.yaml")
    device = torch.device("cuda")

    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    geom_loss = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    sigmas = [0.01, 0.02, 0.03]
    rows = []  # (sigma, vd_b, igsd_b, vd_o, igsd_o)

    for sigma in sigmas:
        print(f"\n{'='*65}")
        print(f"  sigma = {sigma}")
        print(f"{'='*65}")

        res_tag = args.resolution or "all"
        print(f"  Loading data (noise={sigma}, res={res_tag})...")
        x_clean, x_noisy = collect_data(
            trainer, cfg, sigma, args.num_shapes, device, args.resolution)
        N = x_clean.shape[0]
        print(f"  {N} patches loaded")

        # P2P-Bridge baseline
        print(f"  [P2P-Bridge] denoising...", end="", flush=True)
        with torch.no_grad():
            res_b = ddpm_denoise(
                trainer.model, trainer.sb_schedule, x_noisy,
                sampling_steps=args.sampling_steps, verbose=False)
        vd_b, igsd_b = evaluate(res_b["x_pred"], x_clean)
        print(f" VD={vd_b.mean():.4f}  IGSD={igsd_b.mean():.6f}")

        # Orth guided
        label = f"Orth-{args.guidance_scale}"
        print(f"  [{label}] denoising...", end="", flush=True)
        res_o = ddpm_denoise_ortho_guided(
            trainer.model, trainer.sb_schedule, x_noisy,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            annealing="linear_decay",
            geom_loss=geom_loss,
            grad_clip=2.0,
            verbose=False)
        vd_o, igsd_o = evaluate(res_o["x_pred"], x_clean)
        print(f" VD={vd_o.mean():.4f}  IGSD={igsd_o.mean():.6f}")

        rows.append((sigma, vd_b, igsd_b, vd_o, igsd_o))

    # ── Summary table ──
    gs = args.guidance_scale
    res_tag = args.resolution or "all resolutions"
    print(f"\n{'='*80}")
    print(f"  P2P-Bridge  vs  Orth-{gs}   ({args.num_shapes} patches, {res_tag})")
    print(f"{'='*80}")
    print(f"{'sigma':>7} | {'VD (P2PB)':>14} {'VD (Orth)':>14} {'Δ VD':>8}"
          f" | {'IGSD (P2PB)':>14} {'IGSD (Orth)':>14} {'Δ IGSD':>8}")
    print("-" * 80)
    for sigma, vd_b, igsd_b, vd_o, igsd_o in rows:
        dvd = (vd_b.mean() - vd_o.mean()) / max(vd_b.mean(), 1e-8) * 100
        digsd = (igsd_b.mean() - igsd_o.mean()) / max(igsd_b.mean(), 1e-8) * 100
        vd_b_s = f"{vd_b.mean():.4f}±{vd_b.std():.4f}"
        vd_o_s = f"{vd_o.mean():.4f}±{vd_o.std():.4f}"
        igsd_b_s = f"{igsd_b.mean():.6f}±{igsd_b.std():.6f}"
        igsd_o_s = f"{igsd_o.mean():.6f}±{igsd_o.std():.6f}"
        print(f"{sigma:>7.2f} | {vd_b_s:>14} {vd_o_s:>14} {dvd:>+7.1f}%"
              f" | {igsd_b_s:>14} {igsd_o_s:>14} {digsd:>+7.1f}%")
    print("=" * 80)
    print("  Δ > 0 → Orth wins  |  Δ < 0 → P2P-Bridge wins")
    print("=" * 80)


if __name__ == "__main__":
    main()
