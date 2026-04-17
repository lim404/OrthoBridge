"""Comprehensive evaluation with industry-standard 3D point cloud metrics.

Computes per-shape and set-level metrics for:
  Baseline, Standard Guidance (multiple lambda), Orthogonal Guidance (multiple lambda)

Metrics:
  Per-shape:  CD (x1000), EMD, VD, IGSD
  Set-level:  1-NNA-CD, 1-NNA-EMD, COV-CD, COV-EMD, MMD-CD, MMD-EMD, JSD

Usage:
  PYTHONPATH=. python experiments/eval_standard_metrics.py [--rerun] [--noise 0.03]
"""
import os
import sys
import json
import numpy as np
import torch
from collections import OrderedDict

# =========================================================================== #
#  Data loading & denoising
# =========================================================================== #

def load_test_shapes(cfg, device, num_shapes=40, seed=0):
    """Load clean test patches with fixed seed."""
    from sb_cover.data.punet_loader import get_punet_loaders
    from models.train_utils import to_cuda

    torch.manual_seed(seed)
    np.random.seed(seed)
    _, test_loader = get_punet_loaders(
        data_dir=cfg.data.data_dir,
        patch_size=cfg.data.get("npoints", 2048),
        batch_size=8, noise_min=0.0, noise_max=0.001,
        num_workers=0, num_patches=10,
    )
    all_clean = []
    n = 0
    for batch in test_loader:
        if n >= num_shapes:
            break
        batch = to_cuda(batch, device)
        x = batch["clean_points"].squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)
        use = min(x.shape[0], num_shapes - n)
        all_clean.append(x[:use])
        n += use
    return torch.cat(all_clean)  # (N, 3, pts)


def add_noise(x_clean, noise_std, seed):
    rng = torch.Generator(device=x_clean.device)
    rng.manual_seed(seed)
    return x_clean + noise_std * torch.randn(
        x_clean.shape, device=x_clean.device, generator=rng)


def run_all_methods(trainer, x_noisy, configs, device):
    """Run baseline + all guided configs, return dict of predictions.

    Args:
        trainer: TrainerIGV with loaded model
        x_noisy: (N, 3, pts) noisy input
        configs: list of (method, lambda, annealing, label) tuples
        device: torch device

    Returns:
        results: dict {label: (N, 3, M) tensor of predictions}
    """
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss,
    )

    gl = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    N = x_noisy.shape[0]
    B = 8
    results = {}

    # Baseline
    print("  [Baseline]...", end="", flush=True)
    preds = []
    for start in range(0, N, B):
        end = min(start + B, N)
        with torch.no_grad():
            r = ddpm_denoise(trainer.model, trainer.sb_schedule,
                             x_noisy[start:end], sampling_steps=10, verbose=False)
        preds.append(r["x_pred"].detach())
    results["Baseline"] = torch.cat(preds)
    print(" done")

    # Guided methods
    for method, lam, anneal, label in configs:
        print(f"  [{label}]...", end="", flush=True)
        fn = ddpm_denoise_guided if method == "std" else ddpm_denoise_ortho_guided
        preds = []
        for start in range(0, N, B):
            end = min(start + B, N)
            r = fn(
                trainer.model, trainer.sb_schedule, x_noisy[start:end],
                sampling_steps=10, guidance_scale=lam,
                annealing=anneal, geom_loss=gl,
                grad_clip=2.0, verbose=False,
            )
            preds.append(r["x_pred"].detach())
        results[label] = torch.cat(preds)
        print(" done")

    return results


# =========================================================================== #
#  Metric computation
# =========================================================================== #

def compute_pershape_cd_emd(pred, gt, batch_size=8):
    """Compute per-shape CD and EMD.

    Args:
        pred: (N, 3, M) predictions
        gt:   (N, 3, P) ground truth

    Returns:
        cd_arr: (N,) CD values (x1000)
        emd_arr: (N,) EMD values
    """
    from metrics.chamfer3D.dist_chamfer_3D import chamfer_3DDist_nograd
    from metrics.PyTorchEMD.emd_nograd import earth_mover_distance_nograd

    def distChamferCUDAnograd(x, y):
        d1, d2, _, _ = chamfer_3DDist_nograd()(x.cuda(), y.cuda())
        return d1, d2

    def emd_approx(x, y):
        return earth_mover_distance_nograd(x.cuda(), y.cuda(), transpose=False)

    # Transpose to (N, M, 3) and (N, P, 3)
    pred_t = pred.transpose(1, 2).contiguous().cuda()
    gt_t = gt.transpose(1, 2).contiguous().cuda()

    N = pred_t.shape[0]
    cd_list, emd_list = [], []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        p_batch = pred_t[start:end]
        g_batch = gt_t[start:end]

        # CD
        dl, dr = distChamferCUDAnograd(p_batch, g_batch)
        cd = (dl.mean(dim=1) + dr.mean(dim=1))  # (B,)
        cd_list.append(cd.cpu())

        # EMD
        emd = emd_approx(p_batch, g_batch)
        emd_list.append(emd.cpu())

    cd_arr = torch.cat(cd_list).numpy() * 1000  # x1000
    emd_arr = torch.cat(emd_list).numpy()
    return cd_arr, emd_arr


def compute_pershape_vd_igsd(pred, gt):
    """Compute per-shape VD and IGSD.

    Args:
        pred: (N, 3, M) predictions
        gt:   (N, 3, P) ground truth

    Returns:
        vd_arr, igsd_arr: (N,) arrays
    """
    from metrics.geometric_metrics import compute_vd, compute_igsd

    N = pred.shape[0]
    vd_arr = np.zeros(N)
    igsd_arr = np.zeros(N)

    for i in range(N):
        p = pred[i].transpose(0, 1).cpu()  # (M, 3)
        g = gt[i].transpose(0, 1).cpu()    # (P, 3)
        vd_arr[i] = compute_vd(p, g)
        igsd_arr[i] = compute_igsd(p, g)

    return vd_arr, igsd_arr


def _pairwise_cd_matrix(pcs_a, pcs_b, batch_size=8):
    """Compute pairwise CD matrix (Na, Nb).

    pcs_a: (Na, M, 3), pcs_b: (Nb, M, 3) on CUDA.
    """
    from metrics.chamfer3D.dist_chamfer_3D import chamfer_3DDist_nograd
    chamfer_fn = chamfer_3DDist_nograd()

    Na, Nb = pcs_a.shape[0], pcs_b.shape[0]
    dist_matrix = torch.zeros(Na, Nb)

    for i in range(Na):
        a_exp = pcs_a[i:i+1].expand(min(batch_size, Nb), -1, -1)
        for j_start in range(0, Nb, batch_size):
            j_end = min(j_start + batch_size, Nb)
            b_batch = pcs_b[j_start:j_end]
            bs = b_batch.shape[0]
            d1, d2, _, _ = chamfer_fn(
                pcs_a[i:i+1].expand(bs, -1, -1).contiguous(),
                b_batch,
            )
            cd = d1.mean(dim=1) + d2.mean(dim=1)  # (bs,)
            dist_matrix[i, j_start:j_end] = cd.cpu()

    return dist_matrix


def _pairwise_emd_matrix(pcs_a, pcs_b, batch_size=4):
    """Compute pairwise EMD matrix (Na, Nb).

    pcs_a: (Na, M, 3), pcs_b: (Nb, M, 3) on CUDA.
    """
    from metrics.PyTorchEMD.emd_nograd import earth_mover_distance_nograd

    Na, Nb = pcs_a.shape[0], pcs_b.shape[0]
    dist_matrix = torch.zeros(Na, Nb)

    for i in range(Na):
        for j_start in range(0, Nb, batch_size):
            j_end = min(j_start + batch_size, Nb)
            b_batch = pcs_b[j_start:j_end]
            bs = b_batch.shape[0]
            emd = earth_mover_distance_nograd(
                pcs_a[i:i+1].expand(bs, -1, -1).contiguous(),
                b_batch,
                transpose=False,
            )
            dist_matrix[i, j_start:j_end] = emd.cpu()

    return dist_matrix


def _knn_accuracy(Mxx, Mxy, Myy, k=1):
    """1-Nearest Neighbor Accuracy (two-sample test).

    Mxx: (N0, N0) ref-ref distances
    Mxy: (N0, N1) ref-sample distances
    Myy: (N1, N1) sample-sample distances

    Returns accuracy (ideal = 50%).
    """
    n0, n1 = Mxx.size(0), Myy.size(0)
    label = torch.cat([torch.ones(n0), torch.zeros(n1)])

    M = torch.cat([
        torch.cat([Mxx, Mxy], dim=1),
        torch.cat([Mxy.t(), Myy], dim=1),
    ], dim=0)

    INF = float('inf')
    M_masked = M + torch.diag(INF * torch.ones(n0 + n1))
    _, idx = M_masked.topk(k, dim=0, largest=False)

    count = torch.zeros(n0 + n1)
    for i in range(k):
        count += label[idx[i]]
    pred = (count >= k / 2.0).float()

    acc = (label == pred).float().mean().item()
    return acc


def _mmd_cov(dist_matrix):
    """Compute MMD and Coverage from sample-to-ref distance matrix.

    dist_matrix: (N_sample, N_ref)
    Returns dict with mmd, cov.
    """
    N_sample, N_ref = dist_matrix.shape
    min_from_sample, min_idx = dist_matrix.min(dim=1)
    min_from_ref, _ = dist_matrix.min(dim=0)

    mmd = min_from_ref.mean().item()
    cov = min_idx.unique().numel() / float(N_ref)
    return {"mmd": mmd, "cov": cov}


def compute_set_metrics(pred, gt, batch_size=8):
    """Compute set-level metrics: 1-NNA, COV, MMD (with CD and EMD).

    Args:
        pred: (N, 3, M) predictions
        gt:   (N, 3, P) ground truth

    Returns:
        dict with 1-NNA-CD, 1-NNA-EMD, COV-CD, COV-EMD, MMD-CD, MMD-EMD
    """
    sample_pcs = pred.transpose(1, 2).contiguous().cuda()
    ref_pcs = gt.transpose(1, 2).contiguous().cuda()

    M = sample_pcs.shape[1]
    P = ref_pcs.shape[1]
    if P != M:
        idx = torch.randperm(P)[:M]
        ref_pcs = ref_pcs[:, idx, :]

    results = {}

    # CD-based metrics
    print("    Pairwise CD (ref×sample)...", end="", flush=True)
    M_rs_cd = _pairwise_cd_matrix(ref_pcs, sample_pcs, batch_size)
    print(" done")
    print("    Pairwise CD (ref×ref)...", end="", flush=True)
    M_rr_cd = _pairwise_cd_matrix(ref_pcs, ref_pcs, batch_size)
    print(" done")
    print("    Pairwise CD (sample×sample)...", end="", flush=True)
    M_ss_cd = _pairwise_cd_matrix(sample_pcs, sample_pcs, batch_size)
    print(" done")

    results["1-NN-CD-acc"] = _knn_accuracy(M_rr_cd, M_rs_cd, M_ss_cd, k=1)
    mc = _mmd_cov(M_rs_cd.t())  # transpose: (sample, ref) -> match sample to ref
    results["lgan_mmd-CD"] = mc["mmd"]
    results["lgan_cov-CD"] = mc["cov"]

    # EMD-based metrics
    print("    Pairwise EMD (ref×sample)...", end="", flush=True)
    M_rs_emd = _pairwise_emd_matrix(ref_pcs, sample_pcs, batch_size=4)
    print(" done")
    print("    Pairwise EMD (ref×ref)...", end="", flush=True)
    M_rr_emd = _pairwise_emd_matrix(ref_pcs, ref_pcs, batch_size=4)
    print(" done")
    print("    Pairwise EMD (sample×sample)...", end="", flush=True)
    M_ss_emd = _pairwise_emd_matrix(sample_pcs, sample_pcs, batch_size=4)
    print(" done")

    results["1-NN-EMD-acc"] = _knn_accuracy(M_rr_emd, M_rs_emd, M_ss_emd, k=1)
    mc_emd = _mmd_cov(M_rs_emd.t())
    results["lgan_mmd-EMD"] = mc_emd["mmd"]
    results["lgan_cov-EMD"] = mc_emd["cov"]

    return results


# =========================================================================== #
#  Main evaluation
# =========================================================================== #

def main():
    from omegaconf import OmegaConf
    from sb_cover.training.trainer_igv import TrainerIGV

    noise_std = 0.03
    seed = 42
    num_shapes = 40

    # Parse args
    if "--noise" in sys.argv:
        idx = sys.argv.index("--noise")
        noise_std = float(sys.argv[idx + 1])

    cache_dir = "experiments/results/standard_metrics"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"predictions_sigma{noise_std:.3f}.pt")

    # Guided configs: (method, lambda, annealing, label)
    configs = [
        ("std",  1.0, "linear_decay", "Std-ld-1.0"),
        ("std",  3.0, "linear_decay", "Std-ld-3.0"),
        ("orth", 1.0, "linear_decay", "Orth-ld-1.0"),
        ("orth", 2.0, "linear_decay", "Orth-ld-2.0"),
        ("orth", 3.0, "linear_decay", "Orth-ld-3.0"),
    ]
    all_labels = ["Baseline"] + [c[3] for c in configs]

    # ── Stage 1: Run denoising or load cache ──
    if os.path.exists(cache_file) and "--rerun" not in sys.argv:
        print(f"Loading cached predictions from {cache_file}")
        cache = torch.load(cache_file, map_location="cpu", weights_only=True)
        x_clean = cache["clean"]
        predictions = cache["predictions"]
    else:
        print("Running denoising experiments...")
        cfg = OmegaConf.load("configs/shapenet_denoise_sb_igv.yaml")
        device = torch.device("cuda")

        trainer = TrainerIGV(cfg, device=device)
        trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
        trainer.model.eval()

        x_clean = load_test_shapes(cfg, device, num_shapes=num_shapes)
        x_noisy = add_noise(x_clean, noise_std, seed)
        print(f"Loaded {x_clean.shape[0]} shapes, noise sigma={noise_std}")

        predictions = run_all_methods(trainer, x_noisy, configs, device)

        # Save cache (move to CPU)
        cache = {
            "clean": x_clean.cpu(),
            "predictions": {k: v.cpu() for k, v in predictions.items()},
        }
        torch.save(cache, cache_file)
        print(f"Saved predictions to {cache_file}")

        x_clean = x_clean.cpu()
        predictions = {k: v.cpu() for k, v in predictions.items()}

    N = x_clean.shape[0]
    npts_clean = x_clean.shape[2]
    npts_pred = predictions["Baseline"].shape[2]
    print(f"\nN={N} shapes, clean={npts_clean} pts, pred={npts_pred} pts")
    print(f"Noise sigma={noise_std}, seed={seed}")

    # ── Stage 2: Per-shape metrics ──
    print("\n" + "=" * 100)
    print("  PER-SHAPE METRICS")
    print("=" * 100)

    pershape = {}
    for label in all_labels:
        print(f"\n  Computing metrics for [{label}]...")
        pred = predictions[label]

        cd, emd = compute_pershape_cd_emd(pred, x_clean)
        vd, igsd = compute_pershape_vd_igsd(pred, x_clean)

        pershape[label] = {
            "CD": cd, "EMD": emd, "VD": vd, "IGSD": igsd,
        }

    # Per-shape results table
    print(f"\n  {'Method':<16} {'CD(x1e3)':>10} {'EMD':>10} "
          f"{'VD':>10} {'IGSD':>12}")
    print(f"  {'-' * 62}")
    base = pershape["Baseline"]
    for label in all_labels:
        m = pershape[label]
        cd_str = f"{m['CD'].mean():.4f}"
        emd_str = f"{m['EMD'].mean():.4f}"
        vd_str = f"{m['VD'].mean():.4f}"
        igsd_str = f"{m['IGSD'].mean():.6f}"

        if label != "Baseline":
            dcd = (m["CD"].mean() - base["CD"].mean()) / base["CD"].mean() * 100
            demd = (m["EMD"].mean() - base["EMD"].mean()) / base["EMD"].mean() * 100
            dvd = (m["VD"].mean() - base["VD"].mean()) / base["VD"].mean() * 100
            digsd = (m["IGSD"].mean() - base["IGSD"].mean()) / base["IGSD"].mean() * 100
            cd_str += f" ({dcd:+.1f}%)"
            emd_str += f" ({demd:+.1f}%)"
            vd_str += f" ({dvd:+.1f}%)"
            igsd_str += f" ({digsd:+.1f}%)"

        print(f"  {label:<16} {cd_str:>18} {emd_str:>16} "
              f"{vd_str:>16} {igsd_str:>18}")

    # ── Stage 3: Set-level metrics ──
    print("\n" + "=" * 100)
    print("  SET-LEVEL METRICS (1-NNA, COV, MMD)")
    print("=" * 100)

    setlevel = {}
    for label in all_labels:
        print(f"\n  Computing set metrics for [{label}]...")
        pred = predictions[label]
        res = compute_set_metrics(pred, x_clean)
        setlevel[label] = res

    # Set-level results table
    print(f"\n  {'Method':<16} {'1NNA-CD':>10} {'1NNA-EMD':>10} "
          f"{'COV-CD':>10} {'COV-EMD':>10} {'MMD-CD':>12} {'MMD-EMD':>12} "
          f"{'JSD':>8}")
    print(f"  {'-' * 100}")
    for label in all_labels:
        r = setlevel[label]
        nna_cd = r.get("1-NN-CD-acc", float('nan')) * 100
        nna_emd = r.get("1-NN-EMD-acc", float('nan')) * 100
        cov_cd = r.get("lgan_cov-CD", float('nan')) * 100
        cov_emd = r.get("lgan_cov-EMD", float('nan')) * 100
        mmd_cd = r.get("lgan_mmd-CD", float('nan')) * 1000
        mmd_emd = r.get("lgan_mmd-EMD", float('nan')) * 100
        jsd = r.get("jsd", float('nan'))
        print(f"  {label:<16} {nna_cd:>9.2f}% {nna_emd:>9.2f}% "
              f"{cov_cd:>9.2f}% {cov_emd:>9.2f}% "
              f"{mmd_cd:>11.4f} {mmd_emd:>11.4f} "
              f"{jsd:>8.4f}")

    # ── Save all results ──
    results_file = os.path.join(
        cache_dir, f"results_sigma{noise_std:.3f}.json")
    save_data = {
        "config": {
            "noise_std": noise_std, "seed": seed, "num_shapes": N,
            "npts_clean": npts_clean, "npts_pred": npts_pred,
        },
        "per_shape": {
            label: {k: v.tolist() for k, v in metrics.items()}
            for label, metrics in pershape.items()
        },
        "set_level": {
            label: {k: float(v) if not isinstance(v, float) else v
                    for k, v in metrics.items()}
            for label, metrics in setlevel.items()
        },
    }
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ── LaTeX table ──
    print("\n" + "=" * 100)
    print("  LATEX TABLE")
    print("=" * 100)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Standard metrics at $\sigma=" + f"{noise_std}" + r"$, $N=" + f"{N}" + r"$ shapes.}")
    print(r"\label{tab:standard_metrics}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l cccc ccc}")
    print(r"\toprule")
    print(r"Method & CD$\downarrow$ & EMD$\downarrow$ & VD$\downarrow$ & IGSD$\downarrow$"
          r" & 1-NNA-CD$\downarrow$ & COV-CD$\uparrow$ & MMD-CD$\downarrow$ \\")
    print(r"\midrule")
    for label in all_labels:
        m = pershape[label]
        r = setlevel[label]
        cd = m["CD"].mean()
        emd = m["EMD"].mean()
        vd = m["VD"].mean()
        igsd = m["IGSD"].mean()
        nna_cd = r.get("1-NN-CD-acc", float('nan')) * 100
        cov_cd = r.get("lgan_cov-CD", float('nan')) * 100
        mmd_cd = r.get("lgan_mmd-CD", float('nan')) * 1000

        latex_label = label.replace("-", " ").replace("_", " ")
        print(f"{latex_label} & {cd:.4f} & {emd:.4f} & {vd:.4f} & {igsd:.6f}"
              f" & {nna_cd:.1f}\\% & {cov_cd:.1f}\\% & {mmd_cd:.4f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
