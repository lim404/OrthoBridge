"""ShapeNet Generalization Evaluation (Stage 3).

Evaluates OrthoBridge and orthogonal guidance variants on full ShapeNet categories
(Airplane, Car, Chair — 14,337 shapes total). The model was trained on PUNet,
so all ShapeNet shapes are unseen data.

Metrics:
  Per-shape: CD (x1000), EMD
  Set-level: 1-NNA-CD, COV-CD, MMD-CD (subsampled to N_sub for tractability)

Usage:
  PYTHONPATH=. python experiments/shapenet_generalization.py
  PYTHONPATH=. python experiments/shapenet_generalization.py --categories 02691156 --noise 0.01 --n_sub 100
"""

import argparse
import csv
import json
import os
import sys
import time

# Ensure project root is on sys.path so sb_cover / models / metrics are importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import Dataset

# =========================================================================== #
#  Constants
# =========================================================================== #

CATEGORIES = {
    "02691156": "Airplane",
    "02958343": "Car",
    "03001627": "Chair",
}

# (name, method_type, lambda, annealing)
METHOD_CONFIGS = [
    ("OrthoBridge",   "baseline", 0.0, None),
    ("Orth-0.1", "orth",     0.1, "linear_decay"),
    ("Orth-0.3", "orth",     0.3, "linear_decay"),
]

SHAPENET_ROOT = "/mnt/a/Users/Administrator/PycharmProjects/ECCV/data/ShapeNet"
RESULTS_DIR = "experiments/results/shapenet_generalization"


# =========================================================================== #
#  ShapeNet Data Loader
# =========================================================================== #

class ShapeNetCategory(Dataset):
    """Load all .npy files from a ShapeNet category directory."""

    def __init__(self, data_root, synset_id, npoints=2048, seed=0):
        self.npoints = npoints
        self.seed = seed

        cat_dir = os.path.join(data_root, synset_id)
        self.files = sorted([
            os.path.join(cat_dir, f)
            for f in os.listdir(cat_dir)
            if f.endswith(".npy")
        ])
        if not self.files:
            raise RuntimeError(f"No .npy files found in {cat_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Load, subsample, and normalize a single shape.

        Returns:
            Tensor of shape (3, npoints) float32, unit-sphere normalized.
        """
        pc = np.load(self.files[idx])  # (8192, 3)

        # Fixed-seed subsample per shape
        rng = np.random.RandomState(self.seed + idx)
        choice = rng.choice(pc.shape[0], self.npoints, replace=False)
        pc = pc[choice]  # (npoints, 3)

        # Unit sphere normalization
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_r = np.sqrt((pc ** 2).sum(axis=1).max())
        if max_r > 1e-8:
            pc = pc / max_r

        return torch.from_numpy(pc.T).float()  # (3, npoints)


def collect_tensors(dataset, device="cpu"):
    """Collect all shapes from a dataset into a single tensor.

    Returns:
        Tensor of shape (N, 3, npoints).
    """
    tensors = []
    for i in range(len(dataset)):
        tensors.append(dataset[i])
    return torch.stack(tensors).to(device)


# =========================================================================== #
#  Noise
# =========================================================================== #

def add_noise(x_clean, noise_std, seed):
    """Add Gaussian noise with a fixed seed."""
    rng = torch.Generator(device=x_clean.device)
    rng.manual_seed(seed)
    return x_clean + noise_std * torch.randn(
        x_clean.shape, device=x_clean.device, generator=rng)


# =========================================================================== #
#  Denoising Pipeline
# =========================================================================== #

def denoise_category(trainer, x_noisy, method_configs, device, batch_size=32):
    """Denoise all shapes with all methods.

    Args:
        trainer: TrainerIGV with loaded model.
        x_noisy: (N, 3, npoints) noisy input tensor on device.
        method_configs: List of (name, type, lambda, annealing).
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        Dict {method_name: (N, 3, npoints) tensor on CPU}.
    """
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_ortho_guided, GeometricQualityLoss,
    )

    gl = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    N = x_noisy.shape[0]
    results = {}

    for name, method_type, lam, annealing in method_configs:
        print(f"    [{name}] denoising {N} shapes...", end="", flush=True)
        t0 = time.time()
        preds = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = x_noisy[start:end].to(device)

            if method_type == "baseline":
                with torch.no_grad():
                    r = ddpm_denoise(
                        trainer.model, trainer.sb_schedule,
                        batch, sampling_steps=10, verbose=False)
                preds.append(r["x_pred"].cpu())
            else:
                r = ddpm_denoise_ortho_guided(
                    trainer.model, trainer.sb_schedule,
                    batch, sampling_steps=10,
                    guidance_scale=lam, annealing=annealing,
                    geom_loss=gl, grad_clip=2.0, verbose=False)
                preds.append(r["x_pred"].cpu())

            # Free GPU memory
            del batch
            torch.cuda.empty_cache()

        results[name] = torch.cat(preds, dim=0)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s)")

    return results


# =========================================================================== #
#  Metrics
# =========================================================================== #

def compute_pershape_metrics(pred, gt, batch_size=8):
    """Compute per-shape CD and EMD.

    Args:
        pred: (N, 3, M) predictions on CPU.
        gt: (N, 3, M) ground truth on CPU.

    Returns:
        cd_arr: (N,) ndarray, CD x1000
        emd_arr: (N,) ndarray, EMD
    """
    from experiments.eval_standard_metrics import compute_pershape_cd_emd
    return compute_pershape_cd_emd(pred, gt, batch_size=batch_size)


def compute_set_level_metrics(pred, gt, n_sub, batch_size=8, seed=0):
    """Compute set-level 1-NNA-CD, COV-CD, MMD-CD on a subsample.

    Args:
        pred: (N, 3, M) predictions on CPU.
        gt: (N, 3, M) ground truth on CPU.
        n_sub: Subsample size for pairwise computation.
        batch_size: Batch size for pairwise distance computation.
        seed: Random seed for subsampling.

    Returns:
        Dict with 1-NNA-CD, COV-CD, MMD-CD.
    """
    from experiments.eval_standard_metrics import (
        _pairwise_cd_matrix, _knn_accuracy, _mmd_cov,
    )

    N = pred.shape[0]
    n_use = min(n_sub, N)

    # Subsample
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, n_use, replace=False)
    pred_sub = pred[idx]
    gt_sub = gt[idx]

    # Transpose to (N, M, 3) and move to GPU
    sample_pcs = pred_sub.transpose(1, 2).contiguous().cuda()
    ref_pcs = gt_sub.transpose(1, 2).contiguous().cuda()

    # Pairwise CD matrices
    print(f"      Pairwise CD (ref x sample, N={n_use})...", end="", flush=True)
    M_rs = _pairwise_cd_matrix(ref_pcs, sample_pcs, batch_size)
    print(" done")
    print(f"      Pairwise CD (ref x ref)...", end="", flush=True)
    M_rr = _pairwise_cd_matrix(ref_pcs, ref_pcs, batch_size)
    print(" done")
    print(f"      Pairwise CD (sample x sample)...", end="", flush=True)
    M_ss = _pairwise_cd_matrix(sample_pcs, sample_pcs, batch_size)
    print(" done")

    nna = _knn_accuracy(M_rr, M_rs, M_ss, k=1)
    mc = _mmd_cov(M_rs.t())  # transpose: (sample, ref)

    del sample_pcs, ref_pcs
    torch.cuda.empty_cache()

    return {
        "1-NNA-CD": nna,
        "COV-CD": mc["cov"],
        "MMD-CD": mc["mmd"],
    }


# =========================================================================== #
#  Per-category evaluation orchestrator
# =========================================================================== #

def evaluate_category(
    trainer, synset_id, cat_name, noise_std, method_configs,
    device, n_sub, batch_size, skip_denoising, pershape_batch_size=8,
):
    """Run full evaluation for one category at one noise level.

    Returns:
        Dict with per-method results containing per-shape and set-level metrics.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_cache = os.path.join(
        RESULTS_DIR, f"predictions_{synset_id}_{noise_std:.3f}.pt")
    result_cache = os.path.join(
        RESULTS_DIR, f"results_{synset_id}_{noise_std:.3f}.json")

    # Check if results already exist
    if os.path.exists(result_cache) and skip_denoising:
        print(f"  Loading cached results from {result_cache}")
        with open(result_cache) as f:
            return json.load(f)

    # Load data
    print(f"  Loading {cat_name} ({synset_id})...")
    dataset = ShapeNetCategory(SHAPENET_ROOT, synset_id, npoints=2048, seed=0)
    x_clean = collect_tensors(dataset, device="cpu")
    N = x_clean.shape[0]
    print(f"  Loaded {N} shapes, shape={x_clean.shape}")

    # Add noise
    x_noisy = add_noise(x_clean, noise_std, seed=42)

    # Denoise (or load cached predictions)
    if os.path.exists(pred_cache):
        print(f"  Loading cached predictions from {pred_cache}")
        cache = torch.load(pred_cache, map_location="cpu", weights_only=True)
        predictions = cache["predictions"]
        x_clean = cache["clean"]
    else:
        # Run denoising
        predictions = denoise_category(
            trainer, x_noisy, method_configs, device, batch_size=batch_size)

        # Save predictions cache
        cache = {
            "clean": x_clean.cpu(),
            "noisy": x_noisy.cpu(),
            "predictions": {k: v.cpu() for k, v in predictions.items()},
        }
        torch.save(cache, pred_cache)
        print(f"  Saved predictions to {pred_cache}")

    # Compute metrics
    all_results = {}
    method_names = [mc[0] for mc in method_configs]

    for method_name in method_names:
        print(f"  [{method_name}] Computing metrics...")
        pred = predictions[method_name]

        # Per-shape CD/EMD
        cd_arr, emd_arr = compute_pershape_metrics(
            pred, x_clean, batch_size=pershape_batch_size)

        # Set-level metrics
        print(f"    Set-level metrics (n_sub={n_sub}):")
        set_metrics = compute_set_level_metrics(
            pred, x_clean, n_sub=n_sub, batch_size=pershape_batch_size, seed=0)

        all_results[method_name] = {
            "per_shape": {
                "CD_mean": float(cd_arr.mean()),
                "CD_std": float(cd_arr.std()),
                "CD_sem": float(cd_arr.std() / np.sqrt(len(cd_arr))),
                "EMD_mean": float(emd_arr.mean()),
                "EMD_std": float(emd_arr.std()),
                "EMD_sem": float(emd_arr.std() / np.sqrt(len(emd_arr))),
                "N": int(len(cd_arr)),
            },
            "set_level": {
                "1-NNA-CD": float(set_metrics["1-NNA-CD"]),
                "COV-CD": float(set_metrics["COV-CD"]),
                "MMD-CD": float(set_metrics["MMD-CD"]),
                "N_sub": n_sub,
            },
        }

    # Save results
    output = {
        "synset_id": synset_id,
        "category": cat_name,
        "noise_std": noise_std,
        "methods": all_results,
    }
    with open(result_cache, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {result_cache}")

    return output


# =========================================================================== #
#  LaTeX table generation
# =========================================================================== #

def generate_latex_table(all_results, noise_levels, method_names, output_path):
    """Generate a LaTeX table summarizing results across categories and noise levels.

    Args:
        all_results: Dict keyed by (synset_id, noise_std) with evaluation results.
        noise_levels: List of noise_std values.
        method_names: List of method names.
        output_path: Path to write .tex file.
    """
    cat_ids = list(CATEGORIES.keys())
    cat_names = list(CATEGORIES.values())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{ShapeNet generalization: model trained on PUNet, evaluated on full ShapeNet categories.}")
    lines.append(r"\label{tab:shapenet_generalization}")

    for noise_std in noise_levels:
        lines.append(r"\resizebox{\textwidth}{!}{")

        # Build column spec: method | (CD EMD 1NNA) per category | aggregate
        ncols = 1 + 3 * (len(cat_ids) + 1)  # +1 for aggregate
        col_spec = "l" + " ccc" * (len(cat_ids) + 1)
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")

        # Header row 1: category names
        header1 = r"\multirow{2}{*}{Method}"
        for cname in cat_names:
            header1 += r" & \multicolumn{3}{c}{" + cname + "}"
        header1 += r" & \multicolumn{3}{c}{Aggregate}"
        header1 += r" \\"
        lines.append(header1)

        # Header row 2: metric names
        header2 = ""
        for _ in range(len(cat_ids) + 1):
            header2 += r" & CD$\downarrow$ & EMD$\downarrow$ & 1NNA$\downarrow$"
        header2 += r" \\"
        lines.append(r"\cmidrule(lr){2-4}" +
                     "".join(r"\cmidrule(lr){" + str(2+3*i) + "-" + str(4+3*i) + "}"
                             for i in range(1, len(cat_ids) + 1)))
        lines.append(header2)
        lines.append(r"\midrule")

        # Data rows
        for method_name in method_names:
            row = method_name.replace("-", " ")

            # Per-category values
            agg_cd, agg_emd, agg_nna = [], [], []
            for sid in cat_ids:
                key = (sid, noise_std)
                if key in all_results:
                    m = all_results[key]["methods"].get(method_name, {})
                    ps = m.get("per_shape", {})
                    sl = m.get("set_level", {})
                    cd_mean = ps.get("CD_mean", float("nan"))
                    cd_sem = ps.get("CD_sem", float("nan"))
                    emd_mean = ps.get("EMD_mean", float("nan"))
                    emd_sem = ps.get("EMD_sem", float("nan"))
                    nna = sl.get("1-NNA-CD", float("nan")) * 100

                    row += f" & {cd_mean:.3f}$\\pm${cd_sem:.3f}"
                    row += f" & {emd_mean:.4f}$\\pm${emd_sem:.4f}"
                    row += f" & {nna:.1f}\\%"

                    agg_cd.append(cd_mean)
                    agg_emd.append(emd_mean)
                    agg_nna.append(nna)
                else:
                    row += " & -- & -- & --"

            # Aggregate
            if agg_cd:
                row += f" & {np.mean(agg_cd):.3f}"
                row += f" & {np.mean(agg_emd):.4f}"
                row += f" & {np.mean(agg_nna):.1f}\\%"
            else:
                row += " & -- & -- & --"

            row += r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append(f"\\\\[2pt] \\small $\\sigma = {noise_std}$")
        lines.append("")

    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"\nLaTeX table saved to {output_path}")
    return tex


def generate_summary_csv(all_results, noise_levels, method_names, output_path):
    """Generate a CSV summary of all results."""
    rows = []
    cat_ids = list(CATEGORIES.keys())

    for noise_std in noise_levels:
        for method_name in method_names:
            for sid in cat_ids:
                key = (sid, noise_std)
                if key not in all_results:
                    continue
                m = all_results[key]["methods"].get(method_name, {})
                ps = m.get("per_shape", {})
                sl = m.get("set_level", {})
                rows.append({
                    "noise_std": noise_std,
                    "method": method_name,
                    "synset_id": sid,
                    "category": CATEGORIES[sid],
                    "N": ps.get("N", 0),
                    "CD_mean": ps.get("CD_mean", ""),
                    "CD_std": ps.get("CD_std", ""),
                    "CD_sem": ps.get("CD_sem", ""),
                    "EMD_mean": ps.get("EMD_mean", ""),
                    "EMD_std": ps.get("EMD_std", ""),
                    "EMD_sem": ps.get("EMD_sem", ""),
                    "1-NNA-CD": sl.get("1-NNA-CD", ""),
                    "COV-CD": sl.get("COV-CD", ""),
                    "MMD-CD": sl.get("MMD-CD", ""),
                    "N_sub": sl.get("N_sub", ""),
                })

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary CSV saved to {output_path}")


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="ShapeNet generalization evaluation (Stage 3)")
    parser.add_argument(
        "--categories", nargs="+", default=list(CATEGORIES.keys()),
        help="Synset IDs to evaluate (default: all 3)")
    parser.add_argument(
        "--noise", nargs="+", type=float, default=[0.01, 0.02, 0.03],
        help="Noise std levels (default: 0.01 0.02 0.03)")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)")
    parser.add_argument(
        "--n_sub", type=int, default=1000,
        help="Subsample size for set-level metrics (default: 1000)")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for denoising (default: 32)")
    parser.add_argument(
        "--skip_denoising", action="store_true",
        help="Skip denoising, only recompute metrics from cached predictions")
    parser.add_argument(
        "--config", type=str, default="configs/shapenet_denoise_sb_igv.yaml",
        help="Model config path")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    print("=" * 80)
    print("  ShapeNet Generalization Evaluation (Stage 3)")
    print("=" * 80)
    print(f"  Categories: {[CATEGORIES.get(s, s) for s in args.categories]}")
    print(f"  Noise levels: {args.noise}")
    print(f"  N_sub: {args.n_sub}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print()

    from omegaconf import OmegaConf
    from sb_cover.training.trainer_igv import TrainerIGV

    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()
    print("Model loaded.\n")

    # Run evaluations
    all_results = {}
    method_names = [mc[0] for mc in METHOD_CONFIGS]

    total_conditions = len(args.categories) * len(args.noise)
    condition_idx = 0

    for synset_id in args.categories:
        cat_name = CATEGORIES.get(synset_id, synset_id)
        for noise_std in args.noise:
            condition_idx += 1
            print("=" * 80)
            print(f"  [{condition_idx}/{total_conditions}] "
                  f"{cat_name} ({synset_id}), sigma={noise_std}")
            print("=" * 80)

            result = evaluate_category(
                trainer=trainer,
                synset_id=synset_id,
                cat_name=cat_name,
                noise_std=noise_std,
                method_configs=METHOD_CONFIGS,
                device=device,
                n_sub=args.n_sub,
                batch_size=args.batch_size,
                skip_denoising=args.skip_denoising,
            )
            all_results[(synset_id, noise_std)] = result

    # Print summary table
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    for noise_std in args.noise:
        print(f"\n  sigma = {noise_std}")
        header = f"  {'Method':<12}"
        for sid in args.categories:
            cname = CATEGORIES.get(sid, sid)[:8]
            header += f" | {cname:>8} CD  {cname:>8} EMD  {cname:>5} 1NNA"
        print(header)
        print("  " + "-" * len(header))

        for method_name in method_names:
            row = f"  {method_name:<12}"
            for sid in args.categories:
                key = (sid, noise_std)
                if key in all_results:
                    m = all_results[key]["methods"].get(method_name, {})
                    ps = m.get("per_shape", {})
                    sl = m.get("set_level", {})
                    cd = ps.get("CD_mean", float("nan"))
                    emd = ps.get("EMD_mean", float("nan"))
                    nna = sl.get("1-NNA-CD", float("nan")) * 100
                    row += f" | {cd:>11.4f}  {emd:>12.5f}  {nna:>8.1f}%"
                else:
                    row += " |          --            --        --"
            print(row)

    # Generate outputs
    generate_summary_csv(
        all_results, args.noise, method_names,
        os.path.join(RESULTS_DIR, "summary.csv"))

    generate_latex_table(
        all_results, args.noise, method_names,
        os.path.join(RESULTS_DIR, "generalization_table.tex"))

    print("\nDone.")


if __name__ == "__main__":
    main()
