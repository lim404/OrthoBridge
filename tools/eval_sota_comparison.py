"""SOTA Comparison Matrix for Orthogonal Geometric Guidance.

Compares against published baselines from the P2P-Bridge (ECCV 2024) paper.
Runs OrthoBridge baseline and OrthoBridge + Orthogonal Guidance using the paper's
full-object evaluation protocol (patch-based denoising on PUNet test set).

Published baselines (from paper Table 1):
    Bilateral, PCNet, DMR, GLR, ScoreDenoise, MAG, PD-Flow, I-PFN, P2P-Bridge

Our methods (applied to P2P-Bridge backbone):
    OrthoBridge (baseline DDPM sampling)
    Orth lambda=1.0 (orthogonal guidance, linear decay)
    Orth lambda=2.0 (orthogonal guidance, linear decay)

Metrics:
    CD  (x10^4) -- Chamfer Distance in unit sphere (matches paper)
    P2M (x10^4) -- Point-to-Mesh distance in unit sphere (matches paper)
    VD          -- Valuation Difference (novel geometric metric)
    IGSD        -- Integral Geometry Signature Distance (novel)

Usage:
    PYTHONPATH=. python experiments/sota_comparison.py
    PYTHONPATH=. python experiments/sota_comparison.py --res 10000 --noise 0.01
    PYTHONPATH=. python experiments/sota_comparison.py --skip_denoising  # table from cache
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


import argparse
import json
import os
import sys

import numpy as np
import pytorch3d.ops
import torch
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from models.evaluation import (
    chamfer_distance_unit_sphere,
    farthest_point_sampling,
    load_off,
    load_xyz,
    normalize_sphere,
    point_mesh_bidir_distance_single_unit_sphere,
)
from utils.utils import NormalizeUnitSphere

# =========================================================================== #
#  Published SOTA results (P2P-Bridge ECCV 2024, Table 1)
#  All values are x10^4
# =========================================================================== #

PUBLISHED = {
    # (method, resolution, noise): (CD, P2M)
    # --- 10k points ---
    ("Bilateral", 10000, 0.01): (3.65, 1.34),
    ("Bilateral", 10000, 0.02): (5.01, 2.02),
    ("Bilateral", 10000, 0.03): (7.00, 3.56),
    ("PCNet", 10000, 0.01): (3.52, 1.15),
    ("PCNet", 10000, 0.02): (7.47, 3.97),
    ("PCNet", 10000, 0.03): (13.1, 8.74),
    ("DMR", 10000, 0.01): (4.48, 1.72),
    ("DMR", 10000, 0.02): (4.98, 2.12),
    ("DMR", 10000, 0.03): (5.89, 2.85),
    ("GLR", 10000, 0.01): (2.96, 1.05),
    ("GLR", 10000, 0.02): (3.77, 1.31),
    ("GLR", 10000, 0.03): (4.91, 2.11),
    ("ScoreDenoise", 10000, 0.01): (2.52, 0.46),
    ("ScoreDenoise", 10000, 0.02): (3.69, 1.07),
    ("ScoreDenoise", 10000, 0.03): (4.71, 1.94),
    ("MAG", 10000, 0.01): (2.50, 0.46),
    ("MAG", 10000, 0.02): (3.63, 1.05),
    ("MAG", 10000, 0.03): (4.69, 1.92),
    ("PD-Flow", 10000, 0.01): (2.13, 0.38),
    ("PD-Flow", 10000, 0.02): (3.25, 1.01),
    ("PD-Flow", 10000, 0.03): (5.19, 2.52),
    ("I-PFN", 10000, 0.01): (2.31, 0.37),
    ("I-PFN", 10000, 0.02): (3.43, 0.90),
    ("I-PFN", 10000, 0.03): (5.49, 2.50),
    ("P2P-Bridge", 10000, 0.01): (2.28, 0.39),
    ("P2P-Bridge", 10000, 0.02): (3.20, 0.81),
    ("P2P-Bridge", 10000, 0.03): (3.99, 1.42),
    # --- 50k points ---
    ("Bilateral", 50000, 0.01): (0.88, 0.23),
    ("Bilateral", 50000, 0.02): (2.38, 1.39),
    ("Bilateral", 50000, 0.03): (6.30, 4.73),
    ("PCNet", 50000, 0.01): (1.05, 0.35),
    ("PCNet", 50000, 0.02): (1.45, 0.61),
    ("PCNet", 50000, 0.03): (2.29, 1.29),
    ("DMR", 50000, 0.01): (1.16, 0.47),
    ("DMR", 50000, 0.02): (1.57, 0.80),
    ("DMR", 50000, 0.03): (2.43, 1.53),
    ("GLR", 50000, 0.01): (0.70, 0.16),
    ("GLR", 50000, 0.02): (1.59, 0.83),
    ("GLR", 50000, 0.03): (3.84, 2.71),
    ("ScoreDenoise", 50000, 0.01): (0.72, 0.15),
    ("ScoreDenoise", 50000, 0.02): (1.29, 0.57),
    ("ScoreDenoise", 50000, 0.03): (1.93, 1.04),
    ("MAG", 50000, 0.01): (0.71, 0.15),
    ("MAG", 50000, 0.02): (1.29, 0.56),
    ("MAG", 50000, 0.03): (1.93, 1.05),
    ("PD-Flow", 50000, 0.01): (0.65, 0.16),
    ("PD-Flow", 50000, 0.02): (1.42, 0.78),
    ("PD-Flow", 50000, 0.03): (3.90, 2.86),
    ("I-PFN", 50000, 0.01): (0.66, 0.12),
    ("I-PFN", 50000, 0.02): (1.05, 0.43),
    ("I-PFN", 50000, 0.03): (2.54, 1.65),
    ("P2P-Bridge", 50000, 0.01): (0.59, 0.09),
    ("P2P-Bridge", 50000, 0.02): (0.90, 0.32),
    ("P2P-Bridge", 50000, 0.03): (1.56, 0.84),
}

PUBLISHED_METHODS = [
    "Bilateral", "PCNet", "DMR", "GLR",
    "ScoreDenoise", "MAG", "PD-Flow", "I-PFN", "P2P-Bridge",
]

# Our methods: (label, guidance_scale, annealing)
# guidance_scale=None means baseline (no guidance)
OUR_CONFIGS = [
    ("OrthoBridge", None, None),
    ("Orth-0.1", 0.1, "linear_decay"),
    ("Orth-0.3", 0.3, "linear_decay"),
    ("Orth-0.5", 0.5, "linear_decay"),
    ("Orth-1.0", 1.0, "linear_decay"),
]


# =========================================================================== #
#  Patch-based denoising for OrthoBridge model
# =========================================================================== #

def patch_based_denoise_sbigv(
    model,
    sb_schedule,
    pcl_noisy,
    patch_size=2048,
    seed_k=3,
    sampling_steps=10,
    guidance_scale=None,
    annealing=None,
    geom_loss=None,
    grad_clip=2.0,
    batch_size=16,
):
    """Patch-based denoising using OrthoBridge model.

    Mirrors the P2P-Bridge evaluate_objects.py patch extraction +
    reassembly pipeline, but substitutes ddpm_denoise (or ortho-guided
    variant) for the model.sample() call.

    Args:
        model: BridgeFlowModel (OrthoBridge).
        sb_schedule: SBSchedule instance.
        pcl_noisy: (N, 3) noisy point cloud on GPU, unit-sphere normalized.
        patch_size: Points per patch (default 2048, matches P2P-Bridge).
        seed_k: FPS oversampling factor (default 3, matches P2P-Bridge).
        sampling_steps: DDPM reverse steps (default 10).
        guidance_scale: If not None, apply orthogonal guidance at this strength.
        annealing: Annealing schedule for guidance lambda.
        geom_loss: GeometricQualityLoss instance (required if guidance).
        grad_clip: Gradient clipping for guidance.
        batch_size: Max patches per GPU batch.

    Returns:
        pcl_denoised: (N, 3) denoised point cloud.
    """
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import ddpm_denoise_ortho_guided

    assert pcl_noisy.dim() == 2, f"Expected (N, 3), got {pcl_noisy.shape}"
    N, d = pcl_noisy.size()
    pcl_batch = pcl_noisy.unsqueeze(0)  # (1, N, 3)

    # FPS seed points + KNN patches (same as evaluate_objects.py:96-98)
    num_seeds = int(seed_k * N / patch_size)
    seed_pnts, _ = farthest_point_sampling(pcl_batch, num_seeds)
    _, _, patches = pytorch3d.ops.knn_points(
        seed_pnts, pcl_batch, K=patch_size, return_nn=True
    )
    patches = patches[0]  # (num_patches, K, 3)

    # Center and scale each patch (same as evaluate_objects.py:103-106)
    centers = patches.mean(dim=1, keepdim=True)  # (P, 1, 3)
    patches = patches - centers
    scale = torch.max(torch.norm(patches, dim=-1))
    patches = patches / scale

    # Denoise patches in batches
    num_patches = patches.shape[0]
    all_denoised = []

    for start in range(0, num_patches, batch_size):
        end = min(start + batch_size, num_patches)
        batch = patches[start:end].transpose(1, 2)  # (B, 3, K)

        if guidance_scale is not None and guidance_scale > 0:
            result = ddpm_denoise_ortho_guided(
                model, sb_schedule, batch,
                sampling_steps=sampling_steps,
                guidance_scale=guidance_scale,
                annealing=annealing,
                geom_loss=geom_loss,
                grad_clip=grad_clip,
                verbose=False,
            )
        else:
            with torch.no_grad():
                result = ddpm_denoise(
                    model, sb_schedule, batch,
                    sampling_steps=sampling_steps,
                    verbose=False,
                )

        denoised = result["x_pred"].transpose(1, 2)  # (B, K, 3)
        all_denoised.append(denoised.detach())

    patches_denoised = torch.cat(all_denoised)  # (P, K, 3)

    # Undo center + scale
    patches_denoised = patches_denoised * scale + centers

    # Reassemble via FPS (same as evaluate_objects.py:118)
    pcl_denoised, _ = farthest_point_sampling(
        patches_denoised.reshape(1, -1, d), N
    )
    pcl_denoised = pcl_denoised[0].squeeze()  # (N, 3)

    return pcl_denoised


# =========================================================================== #
#  Per-shape evaluation
# =========================================================================== #

def evaluate_method_on_condition(
    model,
    sb_schedule,
    data_path,
    dataset_root,
    dataset,
    res,
    noise,
    guidance_scale=None,
    annealing=None,
    geom_loss=None,
    device="cuda",
    sampling_steps=10,
):
    """Evaluate one method on all test shapes for a given (res, noise).

    Uses the same evaluation protocol as the P2P-Bridge paper:
      1. Load noisy .xyz files
      2. Normalize to unit sphere
      3. Patch-based denoise
      4. Un-normalize
      5. Compute CD and P2M in unit sphere (matches paper metrics)
      6. Compute VD and IGSD (novel metrics)

    Returns:
        dict: {shape_name: {"cd": float, "p2m": float, "vd": float, "igsd": float}}
    """
    from metrics.geometric_metrics import compute_vd, compute_igsd

    input_dir = os.path.join(data_path, f"{dataset}_{res}_poisson_{noise}")
    gt_pcl_dir = os.path.join(
        dataset_root, dataset, "pointclouds", "test", f"{res}_poisson"
    )
    gt_mesh_dir = os.path.join(dataset_root, dataset, "meshes", "test")

    if not os.path.isdir(input_dir):
        logger.error(f"Input dir not found: {input_dir}")
        return {}

    # Load GT
    gt_pcls = load_xyz(gt_pcl_dir)
    gt_meshes = load_off(gt_mesh_dir)

    results = {}

    fnames = sorted([f for f in os.listdir(input_dir) if f.endswith(".xyz")])
    for fn in tqdm(fnames, desc=f"  Eval {res}/{noise}"):
        name = fn[:-4]

        # Load and normalize noisy point cloud
        pcl_noisy = torch.FloatTensor(
            np.loadtxt(os.path.join(input_dir, fn))
        )
        pcl_noisy, center, scale_norm = NormalizeUnitSphere.normalize(pcl_noisy)
        pcl_noisy = pcl_noisy.to(device)

        # Denoise
        model.eval()
        pcl_denoised = patch_based_denoise_sbigv(
            model,
            sb_schedule,
            pcl_noisy,
            patch_size=2048,
            seed_k=3,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
            annealing=annealing,
            geom_loss=geom_loss,
        )

        # Un-normalize for metric computation
        pcl_denoised_cpu = pcl_denoised.cpu() * scale_norm + center

        # ── CD (unit sphere, matches paper) ──
        pcl_up = pcl_denoised_cpu.unsqueeze(0).to(device)
        if name not in gt_pcls:
            logger.warning(f"GT not found for {name}, skipping")
            continue
        pcl_gt = gt_pcls[name].unsqueeze(0).to(device)

        cd = chamfer_distance_unit_sphere(pcl_up, pcl_gt)[0].item()

        # ── P2M (unit sphere, matches paper) ──
        if name not in gt_meshes:
            logger.warning(f"Mesh not found for {name}, skipping P2M")
            p2m = float("nan")
        else:
            verts = gt_meshes[name]["verts"].to(device)
            faces = gt_meshes[name]["faces"].to(device)
            p2m = point_mesh_bidir_distance_single_unit_sphere(
                pcl=pcl_up[0], verts=verts, faces=faces
            ).item()

        # ── VD and IGSD (unit sphere normalized) ──
        # Normalize using GT center/scale for consistency
        pcl_gt_norm, gt_center, gt_scale = normalize_sphere(pcl_gt)
        pcl_up_norm = (pcl_up - gt_center) / gt_scale

        vd = compute_vd(pcl_up_norm[0].cpu(), pcl_gt_norm[0].cpu())
        igsd = compute_igsd(pcl_up_norm[0].cpu(), pcl_gt_norm[0].cpu())

        results[name] = {"cd": cd, "p2m": p2m, "vd": vd, "igsd": igsd}

    return results


# =========================================================================== #
#  Table printing and LaTeX generation
# =========================================================================== #

def print_comparison_tables(all_results, resolutions, noises):
    """Print formatted comparison tables to stdout."""
    for res in resolutions:
        print(f"\n{'=' * 110}")
        print(f"  PUNet Test Set — {res:,} points")
        print(f"{'=' * 110}")

        for noise in noises:
            cond = f"{res}_{noise}"
            print(f"\n  --- sigma = {noise} ---")
            print(
                f"  {'Method':<16}"
                f"{'CD':>10} {'P2M':>10}"
                f"{'VD':>10} {'IGSD':>12}"
                f"{'dCD%':>8} {'dP2M%':>8}"
            )
            print(f"  {'-' * 78}")

            # Published baselines
            p2pb_cd, p2pb_p2m = PUBLISHED.get(
                ("P2P-Bridge", res, noise), (None, None)
            )

            for method in PUBLISHED_METHODS:
                key = (method, res, noise)
                if key not in PUBLISHED:
                    continue
                cd_val, p2m_val = PUBLISHED[key]
                print(
                    f"  {method:<16}"
                    f"{cd_val:>10.2f} {p2m_val:>10.2f}"
                    f"{'—':>10} {'—':>12}"
                    f"{'':>8} {'':>8}"
                )

            # Our methods
            print(f"  {'-' * 78}")
            if cond in all_results:
                for label, _, _ in OUR_CONFIGS:
                    if label not in all_results[cond]:
                        continue
                    shape_results = all_results[cond][label]
                    if not shape_results:
                        continue

                    cds = [v["cd"] for v in shape_results.values()]
                    p2ms = [v["p2m"] for v in shape_results.values()]
                    vds = [v["vd"] for v in shape_results.values()]
                    igsds = [v["igsd"] for v in shape_results.values()]

                    cd_mean = np.mean(cds) * 1e4
                    p2m_mean = np.nanmean(p2ms) * 1e4
                    vd_mean = np.mean(vds)
                    igsd_mean = np.mean(igsds)

                    # Relative to P2P-Bridge
                    dcd = ""
                    dp2m = ""
                    if p2pb_cd is not None:
                        dcd = f"{(cd_mean - p2pb_cd) / p2pb_cd * 100:+.1f}%"
                        dp2m = f"{(p2m_mean - p2pb_p2m) / p2pb_p2m * 100:+.1f}%"

                    print(
                        f"  {label:<16}"
                        f"{cd_mean:>10.2f} {p2m_mean:>10.2f}"
                        f"{vd_mean:>10.4f} {igsd_mean:>12.6f}"
                        f"{dcd:>8} {dp2m:>8}"
                    )

    # Cross-noise summary
    print(f"\n\n{'=' * 110}")
    print(f"  CROSS-NOISE SUMMARY (mean across all conditions)")
    print(f"{'=' * 110}")

    for label, _, _ in OUR_CONFIGS:
        cds, p2ms, vds, igsds = [], [], [], []
        p2pb_cds, p2pb_p2ms = [], []
        for res in resolutions:
            for noise in noises:
                cond = f"{res}_{noise}"
                if cond not in all_results or label not in all_results[cond]:
                    continue
                sr = all_results[cond][label]
                if not sr:
                    continue
                cds.extend([v["cd"] * 1e4 for v in sr.values()])
                p2ms.extend([v["p2m"] * 1e4 for v in sr.values()])
                vds.extend([v["vd"] for v in sr.values()])
                igsds.extend([v["igsd"] for v in sr.values()])

                key = ("P2P-Bridge", res, noise)
                if key in PUBLISHED:
                    p2pb_cds.append(PUBLISHED[key][0])
                    p2pb_p2ms.append(PUBLISHED[key][1])

        if cds:
            print(
                f"  {label:<16}"
                f" CD={np.mean(cds):.2f}  P2M={np.nanmean(p2ms):.2f}"
                f"  VD={np.mean(vds):.4f}  IGSD={np.mean(igsds):.6f}"
            )


def generate_latex_table(all_results, resolutions, noises, output_dir):
    """Generate a LaTeX comparison table matching the paper's format."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Point cloud denoising on PUNet. "
        r"CD and P2M ($\times 10^4$, $\downarrow$). "
        r"VD ($\downarrow$) and IGSD ($\downarrow$) are novel geometric "
        r"metrics computed only for our methods. "
        r"Published baselines from \cite{vogel2024p2pbridge}.}"
    )
    lines.append(r"\label{tab:sota}")

    for res in resolutions:
        res_label = f"{res // 1000}k"
        ncols = len(noises)

        # Build column spec
        colspec = "l " + " ".join(["cc"] * ncols)
        lines.append(rf"\resizebox{{\textwidth}}{{!}}{{%")
        lines.append(rf"\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")

        # Header: noise levels
        header = f"Method ({res_label})"
        for noise in noises:
            header += rf" & \multicolumn{{2}}{{c}}{{$\sigma={noise}$}}"
        header += r" \\"
        lines.append(header)

        # Sub-header: CD / P2M
        subheader = ""
        for _ in noises:
            subheader += r" & CD & P2M"
        subheader += r" \\"
        lines.append(subheader)
        lines.append(r"\midrule")

        # Published baselines
        for method in PUBLISHED_METHODS:
            row = method
            for noise in noises:
                key = (method, res, noise)
                if key in PUBLISHED:
                    cd, p2m = PUBLISHED[key]
                    row += f" & {cd:.2f} & {p2m:.2f}"
                else:
                    row += r" & --- & ---"
            row += r" \\"
            # Bold P2P-Bridge (current SOTA)
            if method == "P2P-Bridge":
                lines.append(r"\midrule")
            lines.append(row)

        lines.append(r"\midrule")

        # Our methods
        for label, _, _ in OUR_CONFIGS:
            row = label
            for noise in noises:
                cond = f"{res}_{noise}"
                if cond in all_results and label in all_results[cond]:
                    sr = all_results[cond][label]
                    if sr:
                        cd_mean = np.mean([v["cd"] for v in sr.values()]) * 1e4
                        p2m_mean = np.nanmean(
                            [v["p2m"] for v in sr.values()]
                        ) * 1e4
                        row += f" & {cd_mean:.2f} & {p2m_mean:.2f}"
                    else:
                        row += r" & --- & ---"
                else:
                    row += r" & --- & ---"
            row += r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append("")

    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    path = os.path.join(output_dir, "sota_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {path}")
    print(latex)

    # Also generate a VD/IGSD supplementary table
    _generate_vd_igsd_table(all_results, resolutions, noises, output_dir)


def _generate_vd_igsd_table(all_results, resolutions, noises, output_dir):
    """Generate supplementary table with VD and IGSD metrics."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Geometric quality metrics for our methods on PUNet. "
        r"VD and IGSD ($\downarrow$) measure volumetric fidelity and "
        r"silhouette consistency respectively.}"
    )
    lines.append(r"\label{tab:vd_igsd}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    colspec = "l l " + " ".join(["cc"] * len(noises))
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    header = "Res & Method"
    for noise in noises:
        header += rf" & \multicolumn{{2}}{{c}}{{$\sigma={noise}$}}"
    header += r" \\"
    lines.append(header)

    subheader = " & "
    for _ in noises:
        subheader += r" & VD & IGSD"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    for res in resolutions:
        first = True
        for label, _, _ in OUR_CONFIGS:
            if first:
                row = f"\\multirow{{{len(OUR_CONFIGS)}}}{{*}}{{{res // 1000}k}}"
                first = False
            else:
                row = ""
            row += f" & {label}"

            for noise in noises:
                cond = f"{res}_{noise}"
                if cond in all_results and label in all_results[cond]:
                    sr = all_results[cond][label]
                    if sr:
                        vd = np.mean([v["vd"] for v in sr.values()])
                        igsd = np.mean([v["igsd"] for v in sr.values()])
                        row += f" & {vd:.4f} & {igsd:.6f}"
                    else:
                        row += r" & --- & ---"
                else:
                    row += r" & --- & ---"

            row += r" \\"
            lines.append(row)

        if res != resolutions[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    path = os.path.join(output_dir, "vd_igsd_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"\nVD/IGSD table saved to {path}")
    print(latex)


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="SOTA Comparison Matrix for Orthogonal Geometric Guidance"
    )
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/a/Users/Administrator/PycharmProjects/ECCV/data/objects/examples/",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/a/Users/Administrator/PycharmProjects/ECCV/data/objects/",
    )
    parser.add_argument("--dataset", type=str, default="PUNet")
    parser.add_argument(
        "--res", type=int, nargs="+", default=[10000, 50000]
    )
    parser.add_argument(
        "--noise", type=float, nargs="+", default=[0.01, 0.02, 0.03]
    )
    parser.add_argument(
        "--skip_denoising",
        action="store_true",
        help="Only print tables from cached results",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pretrained/PVDS_PUNet/latest.pth",
        help="Pretrained backbone checkpoint",
    )
    parser.add_argument(
        "--sampling_steps", type=int, default=10,
        help="DDPM reverse sampling steps",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Patch batch size for GPU",
    )
    args = parser.parse_args()

    device = torch.device(args.gpu)
    results_dir = "experiments/results/sota_comparison"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    if not args.skip_denoising:
        # Load OrthoBridge model
        from sb_cover.training.trainer_igv import TrainerIGV
        from sb_cover.evaluation.guided_sampling import GeometricQualityLoss

        cfg = OmegaConf.load("configs/shapenet_denoise_sb_igv.yaml")
        trainer = TrainerIGV(cfg, device=device)
        trainer.load_pretrained_backbone(args.ckpt)
        trainer.model.eval()

        gl = GeometricQualityLoss(
            w_repulsion=1.0,
            w_projection=1.0,
            w_covariance=0,
            subsample_n=512,
        )

        for res in args.res:
            for noise in args.noise:
                cond = f"{res}_{noise}"
                cache_file = os.path.join(results_dir, f"results_{cond}.json")

                # Check cache
                if os.path.exists(cache_file):
                    with open(cache_file) as f:
                        all_results[cond] = json.load(f)
                    logger.info(f"Loaded cached results for {cond}")
                    continue

                logger.info(f"\n{'=' * 80}")
                logger.info(f"  Resolution={res}, Noise={noise}")
                logger.info(f"{'=' * 80}")

                condition_results = {}

                for label, g_scale, anneal in OUR_CONFIGS:
                    logger.info(f"\n  [{label}]")

                    shape_results = evaluate_method_on_condition(
                        trainer.model,
                        trainer.sb_schedule,
                        data_path=args.data_path,
                        dataset_root=args.dataset_root,
                        dataset=args.dataset,
                        res=res,
                        noise=noise,
                        guidance_scale=g_scale,
                        annealing=anneal,
                        geom_loss=gl if g_scale else None,
                        device=device,
                        sampling_steps=args.sampling_steps,
                    )

                    condition_results[label] = shape_results

                    # Print summary
                    if shape_results:
                        cds = [v["cd"] * 1e4 for v in shape_results.values()]
                        p2ms = [
                            v["p2m"] * 1e4
                            for v in shape_results.values()
                            if not np.isnan(v["p2m"])
                        ]
                        vds = [v["vd"] for v in shape_results.values()]
                        igsds = [v["igsd"] for v in shape_results.values()]
                        logger.info(
                            f"  {label}: CD={np.mean(cds):.2f}  "
                            f"P2M={np.mean(p2ms):.2f}  "
                            f"VD={np.mean(vds):.4f}  "
                            f"IGSD={np.mean(igsds):.6f}"
                        )

                all_results[cond] = condition_results

                with open(cache_file, "w") as f:
                    json.dump(condition_results, f, indent=2)
                logger.info(f"Saved to {cache_file}")

    else:
        # Load all cached results
        for res in args.res:
            for noise in args.noise:
                cond = f"{res}_{noise}"
                cache_file = os.path.join(results_dir, f"results_{cond}.json")
                if os.path.exists(cache_file):
                    with open(cache_file) as f:
                        all_results[cond] = json.load(f)
                    logger.info(f"Loaded {cache_file}")
                else:
                    logger.warning(f"No cache for {cond}")

    # Print and generate tables
    print_comparison_tables(all_results, args.res, args.noise)
    generate_latex_table(all_results, args.res, args.noise, results_dir)


if __name__ == "__main__":
    main()
