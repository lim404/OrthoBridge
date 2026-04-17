"""Stage 2: Main Benchmarking — P2P-Bridge vs SB-IGV.

Runs both denoisers on PUNet test set across resolutions and noise levels,
computing CD, P2F, VD, and IGSD metrics. Produces per-shape JSON results,
summary CSV, and LaTeX table for the paper.

Usage:
    python experiments/stage2_benchmarking.py --device cuda:0
    python experiments/stage2_benchmarking.py --skip_p2pb       # SB-IGV only
    python experiments/stage2_benchmarking.py --skip_sbigv      # P2P-Bridge only
    python experiments/stage2_benchmarking.py --resolutions 10000 --noises 0.01  # smoke test
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import omegaconf
import pandas as pd
import torch
from loguru import logger
from torch import Tensor

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.evaluation import (
    chamfer_distance_unit_sphere,
    load_off,
    load_xyz,
    normalize_sphere,
    point_mesh_bidir_distance_single_unit_sphere,
)
from metrics.geometric_metrics import compute_igsd, compute_vd
from utils.utils import NormalizeUnitSphere


# =========================================================================== #
#  Denoiser Protocol
# =========================================================================== #

class Denoiser(Protocol):
    """Abstract interface for point cloud denoisers."""

    def __call__(self, pcl_noisy: Tensor) -> Tensor:
        """Denoise a single point cloud.

        Args:
            pcl_noisy: Noisy point cloud (N, 3) on CUDA, normalized to unit sphere.

        Returns:
            Denoised point cloud (N, 3) on same device.
        """
        ...


# =========================================================================== #
#  P2P-Bridge Denoiser
# =========================================================================== #

class P2PBridgeDenoiser:
    """Wrapper around P2P-Bridge for patch-based denoising."""

    def __init__(
        self,
        model_path: str = "pretrained/PVDS_PUNet/latest.pth",
        device: str = "cuda:0",
        use_ema: bool = True,
        steps: int = 5,
        patch_size: int = 2048,
        seed_k: int = 3,
    ):
        self.device = device
        self.patch_size = patch_size
        self.seed_k = seed_k

        # Load config from checkpoint directory
        cfg_path = os.path.join(os.path.dirname(model_path), "opt.yaml")
        cfg = omegaconf.OmegaConf.load(cfg_path)

        # Merge runtime args
        runtime = omegaconf.OmegaConf.create({
            "model_path": model_path,
            "use_ema": use_ema,
            "steps": steps,
            "local_rank": 0,
            "distribution_type": "none",
            "restart": False,
            "gpu": device,
        })
        cfg = omegaconf.OmegaConf.merge(cfg, runtime)
        self.cfg = cfg

        # Load model
        from models.model_loader import load_diffusion
        self.model, _ = load_diffusion(cfg)
        self.model.eval()
        logger.info("P2P-Bridge loaded from {}", model_path)

    @torch.no_grad()
    def __call__(self, pcl_noisy: Tensor) -> Tensor:
        """Denoise using patch-based approach.

        Args:
            pcl_noisy: (N, 3) normalized point cloud on CUDA.

        Returns:
            (N, 3) denoised point cloud.
        """
        from evaluate_objects import patch_based_denoise

        pcl_denoised, _ = patch_based_denoise(
            model=self.model,
            pcl_noisy=pcl_noisy,
            patch_size=self.patch_size,
            seed_k=self.seed_k,
            cfg=self.cfg,
            save_intermediate=False,
        )
        return pcl_denoised


# =========================================================================== #
#  SB-IGV Denoiser
# =========================================================================== #

class SBIGVDenoiser:
    """Wrapper around SB-IGV for DDPM-based denoising."""

    def __init__(
        self,
        config_path: str = "configs/shapenet_denoise_sb_igv.yaml",
        ckpt_path: str = "checkpoints/sb_igv/epoch_29.pth",
        device: str = "cuda:0",
        sampling_steps: int = 10,
    ):
        self.device_str = device
        self.device = torch.device(device)
        self.sampling_steps = sampling_steps

        # Load config and trainer
        cfg = omegaconf.OmegaConf.load(config_path)
        from sb_cover.training.trainer_igv import TrainerIGV
        self.trainer = TrainerIGV(cfg, self.device)
        self.trainer.load_checkpoint(ckpt_path)
        self.trainer.model.eval()

        self.sb_schedule = self.trainer.sb_schedule
        logger.info("SB-IGV loaded from {}", ckpt_path)

    @torch.no_grad()
    def __call__(self, pcl_noisy: Tensor) -> Tensor:
        """Denoise using DDPM reverse process.

        Args:
            pcl_noisy: (N, 3) normalized point cloud on CUDA.

        Returns:
            (N, 3) denoised point cloud.
        """
        from sb_cover.evaluation.ddpm_sampling import ddpm_denoise

        # ddpm_denoise expects (B, 3, N)
        x_noisy = pcl_noisy.unsqueeze(0).transpose(1, 2)  # (1, 3, N)

        result = ddpm_denoise(
            model=self.trainer.model,
            sb_schedule=self.sb_schedule,
            x_noisy=x_noisy,
            sampling_steps=self.sampling_steps,
            verbose=False,
        )

        # Use coarse backbone prediction (x_pred) after SB posterior iteration.
        # The decoder (dense_pred) is still undertrained and degrades quality;
        # x_pred is the full-cloud output from the DDPM reverse process.
        pcl_denoised = result["x_pred"][0].transpose(0, 1)  # (B,3,N) → (N,3)

        return pcl_denoised


# =========================================================================== #
#  Metric Computation
# =========================================================================== #

@torch.no_grad()
def evaluate_single_shape(
    pcl_denoised: Tensor,
    pcl_gt: Tensor,
    mesh_verts: Optional[Tensor] = None,
    mesh_faces: Optional[Tensor] = None,
) -> Dict[str, float]:
    """Compute all metrics for a single denoised shape.

    Args:
        pcl_denoised: Denoised point cloud (N, 3).
        pcl_gt: Ground truth point cloud (M, 3).
        mesh_verts: GT mesh vertices (V, 3), for P2F.
        mesh_faces: GT mesh faces (F, 3), for P2F.

    Returns:
        Dict with keys: cd, p2f, vd, igsd.
    """
    results = {}

    # CD (x1000) — operates on (B, N, 3) batched format
    cd_val = chamfer_distance_unit_sphere(
        pcl_denoised.unsqueeze(0), pcl_gt.unsqueeze(0)
    )[0].item()
    results["cd"] = cd_val * 1000

    # P2F — needs mesh
    if mesh_verts is not None and mesh_faces is not None:
        try:
            p2f_val = point_mesh_bidir_distance_single_unit_sphere(
                pcl=pcl_denoised, verts=mesh_verts, faces=mesh_faces
            ).item()
            results["p2f"] = p2f_val
        except Exception as e:
            logger.warning("P2F failed: {}", e)
            results["p2f"] = float("nan")
    else:
        results["p2f"] = float("nan")

    # VD — needs normalized coordinates in [-1, 1]
    # normalize_sphere returns (B, N, 3) so we use the GT's normalization
    gt_norm, center, scale = normalize_sphere(pcl_gt.unsqueeze(0))
    gt_norm = gt_norm[0]  # (M, 3)
    pred_norm = (pcl_denoised.unsqueeze(0) - center) / scale
    pred_norm = pred_norm[0]  # (N, 3)
    results["vd"] = compute_vd(pred_norm, gt_norm)

    # IGSD
    results["igsd"] = compute_igsd(pred_norm, gt_norm)

    return results


# =========================================================================== #
#  Benchmark Runner
# =========================================================================== #

def run_benchmark(
    denoiser_name: str,
    denoiser: Denoiser,
    dataset_root: str,
    resolutions: List[int],
    noises: List[float],
    device: str,
    shapes: Optional[List[str]] = None,
) -> Dict:
    """Run benchmark over all (resolution, noise) conditions.

    Args:
        denoiser_name: Name label for this denoiser.
        denoiser: Callable denoiser.
        dataset_root: Root of PUNet data (contains PUNet/ and examples/).
        resolutions: List of resolutions (e.g., [10000, 50000]).
        noises: List of noise levels (e.g., [0.01, 0.02, 0.03]).
        device: CUDA device string.
        shapes: Optional subset of shape names to evaluate.

    Returns:
        Nested dict: results[res][noise][shape_name] = {cd, p2f, vd, igsd}
    """
    # Load GT meshes (shared across all conditions)
    mesh_dir = os.path.join(dataset_root, "PUNet", "meshes", "test")
    logger.info("Loading GT meshes from {}", mesh_dir)
    meshes = load_off(mesh_dir)

    all_results = {}

    for res in resolutions:
        all_results[res] = {}

        # Load GT point clouds for this resolution
        gt_dir = os.path.join(dataset_root, "PUNet", "pointclouds", "test", f"{res}_poisson")
        logger.info("Loading GT point clouds from {}", gt_dir)
        gt_pcls = load_xyz(gt_dir)

        for noise in noises:
            all_results[res][noise] = {}

            # Load noisy point clouds
            noisy_dir = os.path.join(dataset_root, "examples", f"PUNet_{res}_poisson_{noise}")
            if not os.path.isdir(noisy_dir):
                logger.warning("Noisy dir not found: {}, generating noise on-the-fly", noisy_dir)
                noisy_pcls = None
            else:
                logger.info("Loading noisy point clouds from {}", noisy_dir)
                noisy_pcls = load_xyz(noisy_dir)

            shape_names = shapes if shapes else sorted(gt_pcls.keys())
            logger.info(
                "[{}] res={}, noise={}, {} shapes",
                denoiser_name, res, noise, len(shape_names),
            )

            for shape_name in shape_names:
                if shape_name not in gt_pcls:
                    logger.warning("GT not found for {}, skipping", shape_name)
                    continue

                # Get noisy input
                if noisy_pcls is not None and shape_name in noisy_pcls:
                    pcl_noisy_raw = noisy_pcls[shape_name]
                else:
                    # Generate noise on-the-fly
                    pcl_noisy_raw = gt_pcls[shape_name] + noise * torch.randn_like(gt_pcls[shape_name])

                # Normalize to unit sphere
                pcl_noisy_norm, center, scale = NormalizeUnitSphere.normalize(pcl_noisy_raw)
                pcl_noisy_norm = pcl_noisy_norm.to(device)

                # Denoise
                try:
                    pcl_denoised = denoiser(pcl_noisy_norm)
                except Exception as e:
                    logger.error("[{}] Failed on {} (res={}, noise={}): {}",
                                 denoiser_name, shape_name, res, noise, e)
                    all_results[res][noise][shape_name] = {
                        "cd": float("nan"), "p2f": float("nan"),
                        "vd": float("nan"), "igsd": float("nan"),
                    }
                    continue

                # Get GT point cloud on device
                pcl_gt = gt_pcls[shape_name].to(device)

                # Get mesh if available
                mesh_verts = meshes[shape_name]["verts"].to(device) if shape_name in meshes else None
                mesh_faces = meshes[shape_name]["faces"].to(device) if shape_name in meshes else None

                # Compute metrics
                metrics = evaluate_single_shape(
                    pcl_denoised=pcl_denoised,
                    pcl_gt=pcl_gt,
                    mesh_verts=mesh_verts,
                    mesh_faces=mesh_faces,
                )

                all_results[res][noise][shape_name] = metrics
                logger.info(
                    "  {} | CD={:.4f}  P2F={:.6f}  VD={:.4f}  IGSD={:.6f}",
                    shape_name, metrics["cd"], metrics["p2f"],
                    metrics["vd"], metrics["igsd"],
                )

            # Clear CUDA cache between conditions
            torch.cuda.empty_cache()

    return all_results


# =========================================================================== #
#  Aggregation & Output
# =========================================================================== #

def aggregate_results(results: Dict) -> pd.DataFrame:
    """Aggregate per-shape results into mean +/- std summary.

    Args:
        results: Nested dict from run_benchmark.

    Returns:
        DataFrame with columns: res, noise, cd_mean, cd_std, p2f_mean, p2f_std,
        vd_mean, vd_std, igsd_mean, igsd_std.
    """
    rows = []
    for res in sorted(results.keys()):
        for noise in sorted(results[res].keys()):
            shape_metrics = results[res][noise]
            if not shape_metrics:
                continue

            cds = [m["cd"] for m in shape_metrics.values() if not np.isnan(m["cd"])]
            p2fs = [m["p2f"] for m in shape_metrics.values() if not np.isnan(m["p2f"])]
            vds = [m["vd"] for m in shape_metrics.values() if not np.isnan(m["vd"])]
            igsds = [m["igsd"] for m in shape_metrics.values() if not np.isnan(m["igsd"])]

            rows.append({
                "res": res,
                "noise": noise,
                "cd_mean": np.mean(cds) if cds else float("nan"),
                "cd_std": np.std(cds) if cds else float("nan"),
                "p2f_mean": np.mean(p2fs) if p2fs else float("nan"),
                "p2f_std": np.std(p2fs) if p2fs else float("nan"),
                "vd_mean": np.mean(vds) if vds else float("nan"),
                "vd_std": np.std(vds) if vds else float("nan"),
                "igsd_mean": np.mean(igsds) if igsds else float("nan"),
                "igsd_std": np.std(igsds) if igsds else float("nan"),
            })

    return pd.DataFrame(rows)


def save_results(
    results: Dict,
    denoiser_name: str,
    output_dir: str,
) -> str:
    """Save per-shape JSON and summary CSV.

    Args:
        results: Nested dict from run_benchmark.
        denoiser_name: Name label.
        output_dir: Directory for output files.

    Returns:
        Path to summary CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Per-shape JSON (convert numeric keys to strings for JSON)
    json_results = {}
    for res in results:
        json_results[str(res)] = {}
        for noise in results[res]:
            json_results[str(res)][str(noise)] = results[res][noise]

    json_path = os.path.join(output_dir, f"{denoiser_name}_per_shape.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("Saved per-shape results to {}", json_path)

    # Summary CSV
    summary = aggregate_results(results)
    summary.insert(0, "model", denoiser_name)
    csv_path = os.path.join(output_dir, f"{denoiser_name}_summary.csv")
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info("Saved summary to {}", csv_path)

    return csv_path


def generate_latex_table(
    summaries: Dict[str, pd.DataFrame],
    output_path: str,
) -> str:
    """Generate LaTeX table comparing models.

    Args:
        summaries: Dict mapping model name to summary DataFrame.
        output_path: Path for the .tex file.

    Returns:
        LaTeX table string.
    """
    # Combine all summaries
    all_dfs = []
    for name, df in summaries.items():
        df = df.copy()
        df["model"] = name
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)

    # Build table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Point cloud denoising results on PUNet test set. "
        r"CD ($\times 10^3$, $\downarrow$), P2F ($\downarrow$), "
        r"VD ($\downarrow$), IGSD ($\downarrow$).}",
        r"\label{tab:stage2_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll|cccc}",
        r"\toprule",
        r"Resolution / Noise & Model & CD ($\times 10^3$) & P2F & VD & IGSD \\",
        r"\midrule",
    ]

    for (res, noise), group in combined.groupby(["res", "noise"]):
        first_in_group = True
        for _, row in group.iterrows():
            condition = f"{int(res)} / {noise}" if first_in_group else ""
            cd_str = f"{row['cd_mean']:.3f} $\\pm$ {row['cd_std']:.3f}"
            p2f_str = f"{row['p2f_mean']:.4f} $\\pm$ {row['p2f_std']:.4f}"
            vd_str = f"{row['vd_mean']:.4f} $\\pm$ {row['vd_std']:.4f}"
            igsd_str = f"{row['igsd_mean']:.6f} $\\pm$ {row['igsd_std']:.6f}"
            lines.append(
                f"{condition} & {row['model']} & {cd_str} & {p2f_str} & {vd_str} & {igsd_str} \\\\"
            )
            first_in_group = False
        lines.append(r"\midrule")

    # Remove last \midrule, replace with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info("Saved LaTeX table to {}", output_path)

    return latex


# =========================================================================== #
#  Main
# =========================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: P2P-Bridge vs SB-IGV Benchmarking")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dataset_root", type=str,
        default="/mnt/a/Users/Administrator/PycharmProjects/ECCV/data/objects/",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=[10000, 50000],
    )
    parser.add_argument(
        "--noises", type=float, nargs="+", default=[0.01, 0.02, 0.03],
    )
    parser.add_argument("--output_dir", type=str, default="experiments/results/stage2")
    parser.add_argument("--skip_p2pb", action="store_true", help="Skip P2P-Bridge evaluation")
    parser.add_argument("--skip_sbigv", action="store_true", help="Skip SB-IGV evaluation")
    parser.add_argument(
        "--shapes", type=str, nargs="*", default=None,
        help="Subset of shape names to evaluate (default: all)",
    )

    # P2P-Bridge args
    parser.add_argument("--p2pb_model_path", type=str, default="pretrained/PVDS_PUNet/latest.pth")
    parser.add_argument("--p2pb_steps", type=int, default=5)
    parser.add_argument("--p2pb_seed_k", type=int, default=3)

    # SB-IGV args
    parser.add_argument("--sbigv_config", type=str, default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--sbigv_ckpt", type=str, default="checkpoints/sb_igv/epoch_29.pth")
    parser.add_argument("--sbigv_steps", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    summaries = {}

    # --- P2P-Bridge ---
    if not args.skip_p2pb:
        logger.info("=" * 60)
        logger.info("Evaluating P2P-Bridge")
        logger.info("=" * 60)

        p2pb = P2PBridgeDenoiser(
            model_path=args.p2pb_model_path,
            device=args.device,
            use_ema=True,
            steps=args.p2pb_steps,
            seed_k=args.p2pb_seed_k,
        )

        p2pb_results = run_benchmark(
            denoiser_name="P2P-Bridge",
            denoiser=p2pb,
            dataset_root=args.dataset_root,
            resolutions=args.resolutions,
            noises=args.noises,
            device=args.device,
            shapes=args.shapes,
        )

        save_results(p2pb_results, "P2P-Bridge", args.output_dir)
        summaries["P2P-Bridge"] = aggregate_results(p2pb_results)
        summaries["P2P-Bridge"].insert(0, "model", "P2P-Bridge")

        # Free GPU memory
        del p2pb
        torch.cuda.empty_cache()

    # --- SB-IGV ---
    if not args.skip_sbigv:
        logger.info("=" * 60)
        logger.info("Evaluating SB-IGV")
        logger.info("=" * 60)

        sbigv = SBIGVDenoiser(
            config_path=args.sbigv_config,
            ckpt_path=args.sbigv_ckpt,
            device=args.device,
            sampling_steps=args.sbigv_steps,
        )

        sbigv_results = run_benchmark(
            denoiser_name="SB-IGV",
            denoiser=sbigv,
            dataset_root=args.dataset_root,
            resolutions=args.resolutions,
            noises=args.noises,
            device=args.device,
            shapes=args.shapes,
        )

        save_results(sbigv_results, "SB-IGV", args.output_dir)
        summaries["SB-IGV"] = aggregate_results(sbigv_results)
        summaries["SB-IGV"].insert(0, "model", "SB-IGV")

        del sbigv
        torch.cuda.empty_cache()

    # --- Generate comparison table ---
    if summaries:
        latex = generate_latex_table(
            summaries,
            os.path.join(args.output_dir, "comparison_table.tex"),
        )
        logger.info("\nLaTeX table:\n{}", latex)

        # Also save combined CSV
        combined = pd.concat(summaries.values(), ignore_index=True)
        combined_path = os.path.join(args.output_dir, "combined_summary.csv")
        combined.to_csv(combined_path, index=False, float_format="%.6f")
        logger.info("Saved combined summary to {}", combined_path)

        # Print summary to console
        logger.info("\n" + combined.to_string(index=False))

    logger.success("Stage 2 benchmarking complete!")


if __name__ == "__main__":
    main()
