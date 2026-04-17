"""Score-as-Normal mechanism validation.

Verifies that s_t = pred_x0 - x_t aligns with GT mesh surface normals
across reverse steps, noise levels, and curvature regimes.

Produces two figures:
  Fig 1 (main): angle error + score magnitude, split by curvature
  Fig 2 (supp): magnitude-weighted angle (effective alignment)

Usage:
    python eval_score_as_normal.py
    python eval_score_as_normal.py --num_objects 10 --sampling_steps 20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import point_cloud_utils as pcu
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import Tensor

from sb_cover.training.trainer_igv import TrainerIGV
from sb_cover.losses.sb_interpolation import SBSchedule, space_indices


# ──────────────────────────────────────────────────────────────────── #
#  Mesh utilities
# ──────────────────────────────────────────────────────────────────── #

def load_mesh_normalized(off_path: str):
    v, f = pcu.load_mesh_vf(off_path)
    v = torch.tensor(v, dtype=torch.float32)
    f = torch.tensor(f, dtype=torch.long)
    p_max = v.max(dim=0, keepdim=True).values
    p_min = v.min(dim=0, keepdim=True).values
    center = (p_max + p_min) / 2
    v = v - center
    scale = (v ** 2).sum(dim=1, keepdim=True).sqrt().max().item()
    v = v / scale
    return v, f, center, scale


def compute_face_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v1, dim=1)
    return n / n.norm(dim=1, keepdim=True).clamp(min=1e-10)


def compute_face_centers(verts, faces):
    return (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0


def compute_face_curvature(verts, faces):
    """Per-face curvature via normal angular variation at adjacent vertices."""
    V = verts.shape[0]
    fn = compute_face_normals(verts, faces)
    vert_faces = [[] for _ in range(V)]
    for fi in range(faces.shape[0]):
        for vi in faces[fi]:
            vert_faces[vi.item()].append(fi)

    vert_curv = torch.zeros(V)
    for vi in range(V):
        adj = vert_faces[vi]
        if len(adj) < 2:
            continue
        adj_n = fn[adj]
        mean_n = adj_n.mean(0)
        mean_n = mean_n / mean_n.norm().clamp(min=1e-10)
        cos_a = (adj_n * mean_n.unsqueeze(0)).sum(1).clamp(-1, 1)
        vert_curv[vi] = torch.acos(cos_a).mean()

    return (vert_curv[faces[:, 0]] + vert_curv[faces[:, 1]] +
            vert_curv[faces[:, 2]]) / 3.0


def find_nearest_face(points, face_centers, batch_size=2048):
    indices = []
    for i in range(0, points.shape[0], batch_size):
        chunk = points[i:i + batch_size]
        dist = torch.cdist(chunk, face_centers)
        indices.append(dist.argmin(dim=1))
    return torch.cat(indices)


# ──────────────────────────────────────────────────────────────────── #
#  Step-by-step DDPM with score recording
# ──────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def ddpm_record_scores(model, sb_schedule, x_noisy, sampling_steps=10):
    T = sb_schedule.n_timestep
    steps = space_indices(T, sampling_steps + 1)
    B, C, N = x_noisy.shape
    device = x_noisy.device
    model.eval()
    xt = x_noisy.clone()

    steps_reversed = steps[::-1]
    pair_steps = list(zip(steps_reversed[1:], steps_reversed[:-1]))

    recorded = {"steps": [], "scores": [], "x_chain": [], "score_norms": []}

    for prev_step, step in pair_steps:
        step_tensor = torch.full((B,), step, device=device, dtype=torch.long)
        noise_levels = sb_schedule.noise_levels[step_tensor].detach()
        result = model(x=xt, t=None, noise_level=noise_levels,
                       noisy_input=x_noisy, steps=step_tensor, skip_decoder=True)
        pred_x0 = result["pred_x0_coarse"]
        score = pred_x0 - xt  # (B, 3, N)

        # Per-point score magnitude: (B, N)
        score_mag = score.norm(dim=1)

        recorded["steps"].append(step)
        recorded["scores"].append(score.cpu())
        recorded["x_chain"].append(xt.cpu())
        recorded["score_norms"].append(score_mag.cpu())

        xt = sb_schedule.p_posterior(prev_step, step, xt, pred_x0)

    recorded["x_final"] = xt.cpu()
    return recorded


# ──────────────────────────────────────────────────────────────────── #
#  Angle + magnitude computation
# ──────────────────────────────────────────────────────────────────── #

def compute_angle_and_magnitude(score, points, face_centers, face_normals,
                                face_curvature):
    """Returns angles (degrees), score magnitudes, curvature bucket masks."""
    pts = points.T   # (N, 3)
    s = score.T       # (N, 3)
    s_mag = s.norm(dim=1)  # (N,)

    s_norm = s / s_mag.unsqueeze(1).clamp(min=1e-10)
    nearest_fi = find_nearest_face(pts, face_centers)
    nn_normals = face_normals[nearest_fi]
    nn_curv = face_curvature[nearest_fi]

    # Absolute cosine (normal has ± ambiguity)
    cos_angle = (s_norm * nn_normals).sum(dim=1).abs().clamp(0, 1)
    angles = torch.acos(cos_angle).rad2deg().numpy()

    curv_np = nn_curv.numpy()
    t33, t67 = np.percentile(curv_np, 33.3), np.percentile(curv_np, 66.7)

    return {
        "angles": angles,
        "magnitudes": s_mag.numpy(),
        "low_mask": curv_np <= t33,
        "mid_mask": (curv_np > t33) & (curv_np <= t67),
        "high_mask": curv_np > t67,
    }


# ──────────────────────────────────────────────────────────────────── #
#  Main
# ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_objects", type=int, default=10)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.01, 0.02, 0.03])
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/PVDS_PUNet/latest.pth")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--out_dir", type=str,
                        default="figures/score_as_normal")
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
    print(f"Objects ({len(obj_names)}): {obj_names}\n")

    # Accumulators: [sigma][bucket][step] → list of arrays
    angle_acc = {s: {b: defaultdict(list) for b in
                     ["all", "low", "mid", "high"]}
                 for s in args.sigmas}
    mag_acc = {s: defaultdict(list) for s in args.sigmas}

    for oi, name in enumerate(obj_names):
        print(f"[{oi+1}/{len(obj_names)}] {name}")
        v, f, center, scale = load_mesh_normalized(
            os.path.join(mesh_dir, name + ".off"))
        fn = compute_face_normals(v, f)
        fc = compute_face_centers(v, f)
        fk = compute_face_curvature(v, f)
        print(f"  mesh: {v.shape[0]}V / {f.shape[0]}F")

        pcl = torch.tensor(np.loadtxt(
            os.path.join(pcl_dir, name + ".xyz"), dtype=np.float32))
        pcl = (pcl - center) / scale
        idx = torch.randperm(pcl.shape[0])[:args.npoints]
        pcl_sub = pcl[idx]

        for sigma in args.sigmas:
            noisy = pcl_sub + torch.randn_like(pcl_sub) * sigma
            x_noisy = noisy.T.unsqueeze(0).to(device)

            rec = ddpm_record_scores(trainer.model, trainer.sb_schedule,
                                     x_noisy, args.sampling_steps)

            for step, score, xt in zip(rec["steps"], rec["scores"],
                                       rec["x_chain"]):
                info = compute_angle_and_magnitude(
                    score[0], xt[0], fc, fn, fk)

                for bucket, mask_key in [("all", None), ("low", "low_mask"),
                                         ("mid", "mid_mask"),
                                         ("high", "high_mask")]:
                    a = info["angles"] if mask_key is None \
                        else info["angles"][info[mask_key]]
                    angle_acc[sigma][bucket][step].append(a)

                mag_acc[sigma][step].append(info["magnitudes"])

            final_ang = np.mean(np.concatenate(
                angle_acc[sigma]["all"][rec["steps"][-1]]))
            print(f"  σ={sigma:.2f}  last-step angle={final_ang:.1f}°")

    # ── Aggregate ──
    all_steps = sorted(
        set(s for sig in args.sigmas
            for s in angle_acc[sig]["all"].keys()), reverse=True)
    n_steps = len(all_steps)
    step_x = np.arange(1, n_steps + 1)
    # Also store actual timestep values for annotation
    timestep_values = np.array(all_steps)

    def agg(sigma, bucket):
        means, stds = [], []
        for step in all_steps:
            arrs = angle_acc[sigma][bucket].get(step, [])
            if arrs:
                c = np.concatenate(arrs)
                means.append(np.mean(c))
                stds.append(np.std(c))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        return np.array(means), np.array(stds)

    def agg_mag(sigma):
        means = []
        for step in all_steps:
            arrs = mag_acc[sigma].get(step, [])
            if arrs:
                means.append(np.mean(np.concatenate(arrs)))
            else:
                means.append(np.nan)
        return np.array(means)

    # ================================================================ #
    #  Figure: 2-row × 4-col
    #    Row 1: Angle error (+ random baseline + magnitude)
    #    Row 2: Magnitude-weighted alignment cos|θ|
    # ================================================================ #

    RANDOM_BASELINE = 57.3  # E[arccos(|cos θ|)] for random 3D vectors = 1 rad

    colors = {0.01: "#2196F3", 0.02: "#FF9800", 0.03: "#F44336"}
    buckets = ["all", "low", "mid", "high"]
    titles = ["All Points", "Low Curvature", "Mid Curvature", "High Curvature"]

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.25,
                           height_ratios=[1, 0.8])

    # ── Row 1: Angle error ──
    axes_angle = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for ax, bucket, title in zip(axes_angle, buckets, titles):
        for sigma in args.sigmas:
            m, s = agg(sigma, bucket)
            c = colors[sigma]
            ax.plot(step_x, m, marker="o", ms=3, lw=2,
                    label=f"σ={sigma}", color=c)
            ax.fill_between(step_x, m - s, m + s, alpha=0.12, color=c)

        ax.axhline(RANDOM_BASELINE, ls="--", color="gray", lw=1.2,
                   label="Random (57.3°)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Reverse Step (noisy → clean)", fontsize=10)
        ax.set_xlim(0.5, n_steps + 0.5)
        ax.set_ylim(20, 80)
        ax.grid(True, alpha=0.25)

        # Add timestep annotations on top x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_positions = [1, n_steps // 4, n_steps // 2,
                          3 * n_steps // 4, n_steps]
        tick_labels = [str(timestep_values[i - 1]) for i in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=8, color="gray")
        if ax == axes_angle[0]:
            ax2.set_xlabel("Timestep t", fontsize=9, color="gray")

    axes_angle[0].set_ylabel("Angle Error (degrees) ↓", fontsize=11)
    axes_angle[-1].legend(fontsize=9, loc="lower right")

    # ── Row 2: Score magnitude ──
    axes_mag = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for ax, bucket, title in zip(axes_mag, buckets, titles):
        # Magnitude subplot: same for all buckets (score is per-point, not
        # bucket-specific, but we show the same curve for context)
        for sigma in args.sigmas:
            mag = agg_mag(sigma)
            c = colors[sigma]
            ax.plot(step_x, mag, marker="s", ms=3, lw=2,
                    label=f"σ={sigma}", color=c)

        ax.set_xlabel("Reverse Step (noisy → clean)", fontsize=10)
        ax.set_xlim(0.5, n_steps + 0.5)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"‖s_t‖ Magnitude", fontsize=11, color="gray")

    axes_mag[0].set_ylabel("‖s_t‖ (log scale)", fontsize=11)

    fig.suptitle(
        r"Score-as-Normal Validation: $\angle(s_t,\, n^*)$ and $\|s_t\|$ "
        "across Reverse Process",
        fontsize=15, fontweight="bold", y=1.01)

    out_pdf = os.path.join(args.out_dir, "score_as_normal_validation.pdf")
    out_png = out_pdf.replace(".pdf", ".png")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved: {out_pdf}")

    # ================================================================ #
    #  Figure 2: Effective alignment = ||s_t|| × cos|θ|
    #  Shows magnitude-weighted normal projection → the "useful" component
    # ================================================================ #

    # Accumulate effective alignment
    eff_acc = {s: {b: defaultdict(list) for b in buckets} for s in args.sigmas}
    for sigma in args.sigmas:
        for bucket in buckets:
            for step in all_steps:
                ang_arrs = angle_acc[sigma][bucket].get(step, [])
                mag_arrs = mag_acc[sigma].get(step, [])
                if ang_arrs and mag_arrs:
                    # Match sizes: angles are per-bucket, mags are all-points
                    # Use all-points mean magnitude as normalizer
                    mean_mag = np.mean(np.concatenate(mag_arrs))
                    mean_cos = np.mean(np.abs(np.cos(
                        np.deg2rad(np.concatenate(ang_arrs)))))
                    eff_acc[sigma][bucket][step] = mean_mag * mean_cos

    fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5), sharey=True)

    for ax, bucket, title in zip(axes2, buckets, titles):
        for sigma in args.sigmas:
            vals = [eff_acc[sigma][bucket].get(step, np.nan)
                    for step in all_steps]
            c = colors[sigma]
            ax.plot(step_x, vals, marker="D", ms=3, lw=2,
                    label=f"σ={sigma}", color=c)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Reverse Step (noisy → clean)", fontsize=10)
        ax.set_xlim(0.5, n_steps + 0.5)
        ax.grid(True, alpha=0.25)

    axes2[0].set_ylabel(r"$\|s_t\| \cdot |\cos\theta|$  (effective alignment)",
                        fontsize=11)
    axes2[-1].legend(fontsize=10)
    fig2.suptitle(
        "Effective Normal Alignment: magnitude-weighted projection "
        r"$\|s_t\|\,|\cos\angle(s_t, n^*)|$",
        fontsize=14, fontweight="bold", y=1.02)
    fig2.tight_layout()

    out2_pdf = os.path.join(args.out_dir, "effective_alignment.pdf")
    out2_png = out2_pdf.replace(".pdf", ".png")
    fig2.savefig(out2_pdf, dpi=200, bbox_inches="tight")
    fig2.savefig(out2_png, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {out2_pdf}")

    # ── Numerical summary ──
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nRandom baseline: {RANDOM_BASELINE:.1f}° "
          "(expected angle for random 3D unit vectors)\n")
    print(f"{'σ':>5} {'Bucket':>8} {'Step1':>8} {'Mid':>8} {'Last':>8}"
          f"  {'Δ vs random (step1)':>20}")
    print("-" * 80)
    for sigma in args.sigmas:
        for bucket in buckets:
            m, _ = agg(sigma, bucket)
            mid = len(m) // 2
            delta = RANDOM_BASELINE - m[0]
            print(f"{sigma:>5.2f} {bucket:>8} {m[0]:>7.1f}° {m[mid]:>7.1f}°"
                  f" {m[-1]:>7.1f}°  {delta:>+18.1f}° "
                  f"({'aligned' if delta > 5 else 'weak'})")
        print("-" * 80)

    print(f"\n‖s_t‖ magnitude (all points):")
    for sigma in args.sigmas:
        mag = agg_mag(sigma)
        print(f"  σ={sigma}: step1={mag[0]:.4f}  mid={mag[len(mag)//2]:.4f}"
              f"  last={mag[-1]:.4f}  ratio={mag[0]/mag[-1]:.1f}×")

    # ── Save JSON ──
    numerical = {}
    for sigma in args.sigmas:
        numerical[str(sigma)] = {}
        for bucket in buckets:
            m, s = agg(sigma, bucket)
            numerical[str(sigma)][bucket] = {
                "steps": all_steps,
                "mean_angle": m.tolist(),
                "std_angle": s.tolist(),
            }
        mag = agg_mag(sigma)
        numerical[str(sigma)]["magnitude"] = {
            "steps": all_steps,
            "mean_magnitude": mag.tolist(),
        }
    with open(os.path.join(args.out_dir, "score_as_normal_data.json"), "w") as f:
        json.dump(numerical, f, indent=2)
    print(f"\nData saved to {args.out_dir}/score_as_normal_data.json")


if __name__ == "__main__":
    main()
