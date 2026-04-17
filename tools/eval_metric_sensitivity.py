"""Stage 1: Metric Sensitivity Analysis — Motivation Experiment.

Demonstrates the "blind spots" of Chamfer Distance (CD) and validates
the necessity of VD and IGSD by applying controlled synthetic corruptions
to ground-truth point clouds and measuring metric responses.

Corruptions
-----------
A) Global Shrinkage : scale factors s in [0.85, 0.98]
B) Topological Breaks: drop 5%, 10%, 15% of locally connected patches

Expected result
---------------
CD remains relatively flat (insensitive) under both corruptions,
while VD spikes under shrinkage and IGSD spikes under topological breaks.

Usage
-----
    # With real PUNet data
    python experiments/stage1_metric_sensitivity.py \\
        --data_dir /path/to/data/objects

    # Without data (generates synthetic sphere/airplane-like shapes)
    python experiments/stage1_metric_sensitivity.py --synthetic

    # Custom output directory
    python experiments/stage1_metric_sensitivity.py --synthetic \\
        --output_dir experiments/figures
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.geometric_metrics import ValuationDifference, IntegralGeometrySignatureDistance


# =========================================================================== #
#  Synthetic GT generation (fallback when no real data is available)
# =========================================================================== #

def _fibonacci_sphere(N: int) -> torch.Tensor:
    """Unit-sphere point cloud via Fibonacci lattice (N, 3)."""
    indices = torch.arange(N, dtype=torch.float32)
    phi = torch.acos(1.0 - 2.0 * (indices + 0.5) / N)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    theta = 2.0 * torch.pi * indices / golden
    return torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi),
    ], dim=-1)


def _make_ellipsoid(N: int, a: float, b: float, c: float) -> torch.Tensor:
    """Ellipsoid surface point cloud."""
    pts = _fibonacci_sphere(N)
    pts[:, 0] *= a
    pts[:, 1] *= b
    pts[:, 2] *= c
    return pts


def _make_torus(N: int, R: float = 0.6, r: float = 0.25) -> torch.Tensor:
    """Torus surface point cloud (has non-trivial topology)."""
    theta = torch.linspace(0, 2 * torch.pi, int(N ** 0.5) + 1)[:-1]
    phi = torch.linspace(0, 2 * torch.pi, int(N ** 0.5) + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    theta, phi = theta.flatten(), phi.flatten()
    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)
    pts = torch.stack([x, y, z], dim=-1)
    # Subsample/supersample to exact N
    if pts.shape[0] > N:
        idx = torch.randperm(pts.shape[0])[:N]
        pts = pts[idx]
    elif pts.shape[0] < N:
        extra = pts[torch.randint(0, pts.shape[0], (N - pts.shape[0],))]
        pts = torch.cat([pts, extra], dim=0)
    return pts


def generate_synthetic_gt(num_shapes: int = 100, N: int = 2048) -> list:
    """Generate diverse synthetic GT shapes for sensitivity analysis."""
    shapes = []
    rng = np.random.RandomState(42)

    for i in range(num_shapes):
        shape_type = i % 4
        if shape_type == 0:
            # Sphere with random radius
            r = 0.5 + rng.random() * 0.5
            pts = _fibonacci_sphere(N) * r
        elif shape_type == 1:
            # Ellipsoid
            a = 0.3 + rng.random() * 0.7
            b = 0.3 + rng.random() * 0.7
            c = 0.3 + rng.random() * 0.7
            pts = _make_ellipsoid(N, a, b, c)
        elif shape_type == 2:
            # Torus (topology-sensitive)
            R = 0.4 + rng.random() * 0.3
            r = 0.1 + rng.random() * 0.2
            pts = _make_torus(N, R, r)
        else:
            # Thin elongated shape (sensitive to shrinkage + breaks)
            pts = _fibonacci_sphere(N)
            pts[:, 0] *= 0.15 + rng.random() * 0.15  # thin
            pts[:, 1] *= 0.15 + rng.random() * 0.15
            pts[:, 2] *= 0.6 + rng.random() * 0.4    # elongated

        # Normalize to [-1, 1]
        pts = pts - pts.mean(dim=0)
        max_norm = pts.norm(dim=1).max()
        if max_norm > 1e-6:
            pts = pts / max_norm
        shapes.append(pts)

    return shapes


# =========================================================================== #
#  Real data loading
# =========================================================================== #

def load_punet_gt(data_dir: str, max_shapes: int = 100) -> list:
    """Load GT point clouds from PUNet test set.

    Tries multiple resolutions and collects up to max_shapes.
    """
    shapes = []
    resolutions = ["10000_poisson", "30000_poisson", "50000_poisson"]

    for res in resolutions:
        pcl_dir = os.path.join(data_dir, "PUNet", "pointclouds", "test", res)
        if not os.path.isdir(pcl_dir):
            continue
        for fn in sorted(os.listdir(pcl_dir)):
            if not fn.endswith(".xyz"):
                continue
            pcl = torch.FloatTensor(
                np.loadtxt(os.path.join(pcl_dir, fn), dtype=np.float32)
            )
            # Normalize to unit sphere
            pcl = pcl - pcl.mean(dim=0)
            max_norm = pcl.norm(dim=1).max()
            if max_norm > 1e-6:
                pcl = pcl / max_norm
            shapes.append(pcl)
            if len(shapes) >= max_shapes:
                return shapes

    if not shapes:
        raise FileNotFoundError(
            f"No .xyz files found under {data_dir}/PUNet/pointclouds/test/. "
            "Use --synthetic to run with generated shapes."
        )
    return shapes


# =========================================================================== #
#  Corruption functions
# =========================================================================== #

def corrupt_shrinkage(pcl: torch.Tensor, scale: float) -> torch.Tensor:
    """Corruption A: Global shrinkage towards centroid.

    Mimics mean-reverting artifacts in diffusion models.

    Parameters
    ----------
    pcl : Tensor (N, 3)
        Clean point cloud.
    scale : float
        Scale factor in (0, 1]. 1.0 = no corruption.
    """
    centroid = pcl.mean(dim=0, keepdim=True)
    return centroid + (pcl - centroid) * scale


def corrupt_topology(
    pcl: torch.Tensor,
    drop_ratio: float,
    num_patches: int = 3,
) -> torch.Tensor:
    """Corruption B: Drop locally connected patches (topological breaks).

    Simulates structural disconnection by removing K-NN neighbourhoods
    around randomly chosen seed points.  The returned cloud has fewer
    points (N' = N - n_dropped); all three metrics (CD, VD, IGSD)
    handle different sizes correctly.

    Parameters
    ----------
    pcl : Tensor (N, 3)
        Clean point cloud.
    drop_ratio : float
        Fraction of points to drop (e.g. 0.05, 0.10, 0.15).
    num_patches : int
        Number of seed points (patches) to remove.

    Returns
    -------
    Tensor (N', 3)
        Corrupted point cloud with local patches removed (same device as input).
    """
    if drop_ratio <= 0:
        return pcl.clone()

    N = pcl.shape[0]
    n_drop = int(N * drop_ratio)
    if n_drop == 0:
        return pcl.clone()

    # Run patch selection on CPU to avoid CUDA indexing issues
    orig_device = pcl.device
    pcl_cpu = pcl.detach().cpu()

    k_per_patch = max(n_drop // num_patches, 1)
    mask = torch.ones(N, dtype=torch.bool)

    for _ in range(num_patches):
        available = torch.where(mask)[0]
        if available.numel() == 0:
            break
        seed_idx = available[torch.randint(0, available.numel(), (1,))]
        seed_pt = pcl_cpu[seed_idx]  # (1, 3)

        # Find k nearest neighbours among available points
        dists = (pcl_cpu[available] - seed_pt).norm(dim=1)
        k = min(k_per_patch, available.numel())
        _, nn_local = dists.topk(k, largest=False)
        nn_global = available[nn_local]
        mask[nn_global] = False

    return pcl_cpu[mask].to(orig_device)


# =========================================================================== #
#  Metric computation
# =========================================================================== #

def compute_cd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Chamfer Distance (mean bidirectional nearest-neighbour L2)."""
    dist = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    cd_p2g = dist.min(dim=1).values.mean()
    cd_g2p = dist.min(dim=0).values.mean()
    return (cd_p2g + cd_g2p).item()


# =========================================================================== #
#  Experiment runner
# =========================================================================== #

def run_shrinkage_experiment(
    shapes: list,
    scale_factors: list,
    device: torch.device,
) -> dict:
    """Corruption A: measure CD, VD, IGSD under progressive shrinkage."""
    vd_metric = ValuationDifference(grid_resolution=64, sigma=0.05)
    igsd_metric = IntegralGeometrySignatureDistance(num_directions=128)

    results = {s: {"cd": [], "vd": [], "igsd": []} for s in scale_factors}

    for pcl in tqdm(shapes, desc="Shrinkage experiment"):
        pcl = pcl.to(device)
        for s in scale_factors:
            corrupted = corrupt_shrinkage(pcl, s)
            results[s]["cd"].append(compute_cd(corrupted, pcl))
            results[s]["vd"].append(vd_metric(corrupted, pcl))
            results[s]["igsd"].append(igsd_metric(corrupted, pcl))

    # Average across shapes
    avg = {}
    for s in scale_factors:
        avg[s] = {
            "cd": np.mean(results[s]["cd"]),
            "cd_std": np.std(results[s]["cd"]),
            "vd": np.mean(results[s]["vd"]),
            "vd_std": np.std(results[s]["vd"]),
            "igsd": np.mean(results[s]["igsd"]),
            "igsd_std": np.std(results[s]["igsd"]),
        }
    return avg


def run_topology_experiment(
    shapes: list,
    drop_ratios: list,
    device: torch.device,
) -> dict:
    """Corruption B: measure CD, VD, IGSD under topological breaks."""
    vd_metric = ValuationDifference(grid_resolution=64, sigma=0.05)
    igsd_metric = IntegralGeometrySignatureDistance(num_directions=128)

    results = {d: {"cd": [], "vd": [], "igsd": []} for d in drop_ratios}

    for pcl in tqdm(shapes, desc="Topology experiment"):
        pcl = pcl.to(device)
        for d in drop_ratios:
            torch.manual_seed(42)  # reproducible patch selection
            corrupted = corrupt_topology(pcl, d, num_patches=3)
            corrupted = corrupted.to(device)
            results[d]["cd"].append(compute_cd(corrupted, pcl))
            results[d]["vd"].append(vd_metric(corrupted, pcl))
            results[d]["igsd"].append(igsd_metric(corrupted, pcl))

    avg = {}
    for d in drop_ratios:
        avg[d] = {
            "cd": np.mean(results[d]["cd"]),
            "cd_std": np.std(results[d]["cd"]),
            "vd": np.mean(results[d]["vd"]),
            "vd_std": np.std(results[d]["vd"]),
            "igsd": np.mean(results[d]["igsd"]),
            "igsd_std": np.std(results[d]["igsd"]),
        }
    return avg


# =========================================================================== #
#  Plotting
# =========================================================================== #

def plot_sensitivity(
    shrinkage_results: dict,
    topology_results: dict,
    output_dir: str,
):
    """Generate publication-quality sensitivity analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # ------ Style ------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    colors = {
        "cd":   "#2196F3",   # blue
        "vd":   "#F44336",   # red
        "igsd": "#4CAF50",   # green
    }
    markers = {"cd": "o", "vd": "s", "igsd": "^"}

    # ===================== Figure 1: Shrinkage =====================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    scales = sorted(shrinkage_results.keys(), reverse=True)
    x_labels = [f"{s:.2f}" for s in scales]
    shrinkage_pct = [(1 - s) * 100 for s in scales]  # % shrinkage

    for ax, metric_key, title, ylabel in zip(
        axes,
        ["cd", "vd", "igsd"],
        ["Chamfer Distance (CD)", "Valuation Difference (VD)", "IGSD"],
        ["CD", "VD", "IGSD"],
    ):
        means = [shrinkage_results[s][metric_key] for s in scales]
        stds = [shrinkage_results[s][f"{metric_key}_std"] for s in scales]

        ax.errorbar(
            shrinkage_pct, means, yerr=stds,
            color=colors[metric_key],
            marker=markers[metric_key],
            linewidth=2, markersize=6, capsize=3,
        )
        ax.set_xlabel("Shrinkage (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(
        "Corruption A: Global Shrinkage — Metric Sensitivity",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path_a = os.path.join(output_dir, "stage1_shrinkage_sensitivity.pdf")
    fig.savefig(path_a)
    fig.savefig(path_a.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {path_a}")

    # ===================== Figure 2: Topology =====================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    drops = sorted(topology_results.keys())
    drop_pct = [d * 100 for d in drops]

    for ax, metric_key, title, ylabel in zip(
        axes,
        ["cd", "vd", "igsd"],
        ["Chamfer Distance (CD)", "Valuation Difference (VD)", "IGSD"],
        ["CD", "VD", "IGSD"],
    ):
        means = [topology_results[d][metric_key] for d in drops]
        stds = [topology_results[d][f"{metric_key}_std"] for d in drops]

        ax.errorbar(
            drop_pct, means, yerr=stds,
            color=colors[metric_key],
            marker=markers[metric_key],
            linewidth=2, markersize=6, capsize=3,
        )
        ax.set_xlabel("Patch Drop (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle(
        "Corruption B: Topological Breaks — Metric Sensitivity",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path_b = os.path.join(output_dir, "stage1_topology_sensitivity.pdf")
    fig.savefig(path_b)
    fig.savefig(path_b.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {path_b}")

    # ===================== Figure 3: Combined (paper-ready) =====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # -- Left: Shrinkage --
    ax = axes[0]
    for metric_key, label in [("cd", "CD"), ("vd", "VD"), ("igsd", "IGSD")]:
        means = [shrinkage_results[s][metric_key] for s in scales]
        # Normalize to [0, 1] for fair comparison
        m_min, m_max = min(means), max(means)
        rng = m_max - m_min if m_max > m_min else 1.0
        normed = [(v - m_min) / rng for v in means]
        ax.plot(
            shrinkage_pct, normed,
            color=colors[metric_key],
            marker=markers[metric_key],
            linewidth=2, markersize=6,
            label=label,
        )
    ax.set_xlabel("Shrinkage (%)")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("(a) Global Shrinkage")
    ax.legend()

    # -- Right: Topology --
    ax = axes[1]
    for metric_key, label in [("cd", "CD"), ("vd", "VD"), ("igsd", "IGSD")]:
        means = [topology_results[d][metric_key] for d in drops]
        m_min, m_max = min(means), max(means)
        rng = m_max - m_min if m_max > m_min else 1.0
        normed = [(v - m_min) / rng for v in means]
        ax.plot(
            drop_pct, normed,
            color=colors[metric_key],
            marker=markers[metric_key],
            linewidth=2, markersize=6,
            label=label,
        )
    ax.set_xlabel("Patch Drop (%)")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("(b) Topological Breaks")
    ax.legend()

    fig.suptitle(
        "Metric Sensitivity Analysis: CD vs. VD vs. IGSD",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path_c = os.path.join(output_dir, "stage1_combined_sensitivity.pdf")
    fig.savefig(path_c)
    fig.savefig(path_c.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {path_c}")


# =========================================================================== #
#  CLI
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Metric Sensitivity Analysis")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to data/objects/ directory with PUNet test data.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic shapes instead of real data.")
    p.add_argument("--num_shapes", type=int, default=100,
                   help="Number of GT shapes to use.")
    p.add_argument("--num_points", type=int, default=2048,
                   help="Points per shape (for synthetic).")
    p.add_argument("--output_dir", type=str, default="experiments/figures",
                   help="Output directory for figures.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Compute device (cpu or cuda:0).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Load / generate shapes ---
    if args.synthetic or args.data_dir is None:
        print(f"Generating {args.num_shapes} synthetic shapes "
              f"({args.num_points} points each)...")
        shapes = generate_synthetic_gt(args.num_shapes, args.num_points)
    else:
        print(f"Loading PUNet GT from {args.data_dir}...")
        shapes = load_punet_gt(args.data_dir, max_shapes=args.num_shapes)
    print(f"Loaded {len(shapes)} shapes")

    # --- Corruption A: Global Shrinkage ---
    # Scale factors from 0.98 (mild) down to 0.85 (severe)
    scale_factors = [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.85]
    print("\n=== Corruption A: Global Shrinkage ===")
    shrinkage_results = run_shrinkage_experiment(shapes, scale_factors, device)

    for s in sorted(shrinkage_results.keys(), reverse=True):
        r = shrinkage_results[s]
        print(f"  scale={s:.2f}  CD={r['cd']:.6f}  VD={r['vd']:.4f}  IGSD={r['igsd']:.6f}")

    # --- Corruption B: Topological Breaks ---
    drop_ratios = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    print("\n=== Corruption B: Topological Breaks ===")
    topology_results = run_topology_experiment(shapes, drop_ratios, device)

    for d in sorted(topology_results.keys()):
        r = topology_results[d]
        print(f"  drop={d:.0%}  CD={r['cd']:.6f}  VD={r['vd']:.4f}  IGSD={r['igsd']:.6f}")

    # --- Save raw results ---
    import json
    raw_path = os.path.join(args.output_dir, "stage1_raw_results.json")
    raw = {
        "shrinkage": {str(k): v for k, v in shrinkage_results.items()},
        "topology": {str(k): v for k, v in topology_results.items()},
    }
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")

    # --- Plot ---
    print("\nGenerating figures...")
    plot_sensitivity(shrinkage_results, topology_results, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
