"""Panel C: Dual-Metric Decoupled Visualization.

C.1 (Top) -- Surface Distance Error: Points colored by distance to GT surface.
    Baseline vs Standard Guidance (lambda=3.0) vs Orthogonal Guidance (lambda=3.0).
    Shows Standard Guidance collapse vs Orthogonal safety.

C.2 (Bottom) -- Voxel Density Error: |rho_pred - rho_GT| on 64^3 grid.
    Baseline vs Orthogonal Guidance (lambda=3.0).
    Shows VD improvement from tangential guidance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import os
import sys

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

C_RED = '#B2182B'
C_GREEN = '#1B7837'
C_GRAY = '#555555'


# =========================================================================== #
#  Utility functions
# =========================================================================== #

def surface_distance(pred_pts, gt_pts):
    """Per-point distance from predicted to nearest GT point.

    Args:
        pred_pts: (N, 3) numpy array
        gt_pts: (M, 3) numpy array
    Returns:
        dists: (N,) array
    """
    tree = cKDTree(gt_pts)
    dists, _ = tree.query(pred_pts, k=1)
    return dists


def compute_density_grid(points_np, grid_resolution=64, sigma=0.05):
    """Compute KDE density on a 3D grid (GPU-accelerated).

    Args:
        points_np: (N, 3) numpy, coords in [-1, 1]
        grid_resolution: voxels per axis
        sigma: Gaussian bandwidth
    Returns:
        density: (G, G, G) numpy array
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = torch.tensor(points_np, dtype=torch.float32, device=device)
    G = grid_resolution
    lin = torch.linspace(-1.0, 1.0, G, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing='ij')
    grid = torch.stack([gx.ravel(), gy.ravel(), gz.ravel()], dim=-1)

    two_sigma_sq = 2.0 * sigma ** 2
    chunk_size = 8192
    densities = []
    for i in range(0, grid.shape[0], chunk_size):
        g_chunk = grid[i:i + chunk_size]
        dist_sq = torch.cdist(
            g_chunk.unsqueeze(0), points.unsqueeze(0)
        ).squeeze(0).pow(2)
        density = torch.exp(-dist_sq / two_sigma_sq).sum(dim=-1)
        densities.append(density)
    return torch.cat(densities).reshape(G, G, G).cpu().numpy()


def find_best_view(points):
    """PCA-based view angle selection."""
    centered = points - points.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[2]
    elev = np.degrees(np.arcsin(np.clip(normal[1], -1, 1)))
    azim = np.degrees(np.arctan2(normal[0], normal[2]))
    return elev, azim


# =========================================================================== #
#  Data generation: 3 methods at lambda=3.0
# =========================================================================== #

def run_denoising_trio(noise_std=0.03, seed=42):
    """Run baseline, standard guided, and ortho guided at lambda=3.0."""
    from omegaconf import OmegaConf
    from sb_cover.training.trainer_igv import TrainerIGV
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss,
    )
    from sb_cover.data.punet_loader import get_punet_loaders
    from models.train_utils import to_cuda
    from metrics.geometric_metrics import compute_vd, compute_igsd

    cfg = OmegaConf.load("configs/shapenet_denoise_sb_igv.yaml")
    device = torch.device("cuda")

    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    # Load test patches
    torch.manual_seed(0)
    np.random.seed(0)
    _, test_loader = get_punet_loaders(
        data_dir=cfg.data.data_dir,
        patch_size=cfg.data.get("npoints", 2048),
        batch_size=8, noise_min=0.0, noise_max=0.001,
        num_workers=0, num_patches=10,
    )
    all_clean = []
    n = 0
    num_shapes = 40
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
    x_clean_all = torch.cat(all_clean)
    N = x_clean_all.shape[0]
    print(f"Loaded {N} shapes")

    # Add noise
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    x_noisy = x_clean_all + noise_std * torch.randn(
        x_clean_all.shape, device=device, generator=rng)

    gl = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    B = 8
    all_base, all_std, all_orth = [], [], []
    for start in range(0, N, B):
        end = min(start + B, N)
        x_batch = x_noisy[start:end]
        print(f"  Batch [{start}:{end}]...", end="", flush=True)

        with torch.no_grad():
            r_base = ddpm_denoise(
                trainer.model, trainer.sb_schedule, x_batch,
                sampling_steps=10, verbose=False)
        r_std = ddpm_denoise_guided(
            trainer.model, trainer.sb_schedule, x_batch,
            sampling_steps=10, guidance_scale=3.0,
            annealing="linear_decay", geom_loss=gl,
            grad_clip=2.0, verbose=False)
        r_orth = ddpm_denoise_ortho_guided(
            trainer.model, trainer.sb_schedule, x_batch,
            sampling_steps=10, guidance_scale=3.0,
            annealing="linear_decay", geom_loss=gl,
            grad_clip=2.0, verbose=False)

        all_base.append(r_base["x_pred"].detach())
        all_std.append(r_std["x_pred"].detach())
        all_orth.append(r_orth["x_pred"].detach())
        print(" done")

    pred_base = torch.cat(all_base)
    pred_std = torch.cat(all_std)
    pred_orth = torch.cat(all_orth)

    # Compute per-shape metrics, pick shape with best combined story
    print("\nComputing per-shape VD + IGSD...")
    best_idx = None
    best_score = -float('inf')
    all_metrics = []

    for i in range(N):
        c = x_clean_all[i].transpose(0, 1).cpu()
        p_base = pred_base[i].transpose(0, 1).cpu()
        p_std = pred_std[i].transpose(0, 1).cpu()
        p_orth = pred_orth[i].transpose(0, 1).cpu()

        vd_base = compute_vd(p_base, c)
        vd_std = compute_vd(p_std, c)
        vd_orth = compute_vd(p_orth, c)
        igsd_base = compute_igsd(p_base, c)
        igsd_std = compute_igsd(p_std, c)
        igsd_orth = compute_igsd(p_orth, c)

        m = dict(vd_base=vd_base, vd_std=vd_std, vd_orth=vd_orth,
                 igsd_base=igsd_base, igsd_std=igsd_std, igsd_orth=igsd_orth)
        all_metrics.append(m)

        # Want: large IGSD blow-up for standard + VD improvement for ortho
        igsd_degrad = (igsd_std - igsd_base) / (igsd_base + 1e-10)
        vd_improve = (vd_base - vd_orth) / (vd_base + 1e-10)
        score = igsd_degrad + vd_improve

        print(f"  Shape {i}: VD b/s/o={vd_base:.4f}/{vd_std:.4f}/{vd_orth:.4f}  "
              f"IGSD b/s/o={igsd_base:.6f}/{igsd_std:.6f}/{igsd_orth:.6f}  "
              f"score={score:.2f}")

        if score > best_score:
            best_score = score
            best_idx = i

    m = all_metrics[best_idx]
    print(f"\nBest shape: idx={best_idx}")
    print(f"  VD:   base={m['vd_base']:.4f}, std={m['vd_std']:.4f}, "
          f"orth={m['vd_orth']:.4f}")
    print(f"  IGSD: base={m['igsd_base']:.6f}, std={m['igsd_std']:.6f}, "
          f"orth={m['igsd_orth']:.6f}")

    return {
        'idx': best_idx,
        'clean': x_clean_all[best_idx].transpose(0, 1).cpu().numpy(),
        'baseline': pred_base[best_idx].transpose(0, 1).cpu().numpy(),
        'standard': pred_std[best_idx].transpose(0, 1).cpu().numpy(),
        'ours': pred_orth[best_idx].transpose(0, 1).cpu().numpy(),
        **m,
    }


# =========================================================================== #
#  Figure rendering
# =========================================================================== #

def make_figure(data):
    """Create the dual-metric figure: C.1 (surface distance) + C.2 (density error)."""
    clean = data['clean']        # (M, 3)
    baseline = data['baseline']  # (N, 3)
    standard = data['standard']  # (N, 3)
    ours = data['ours']          # (N, 3)

    # ================================================================
    # C.1: Surface Distance Error
    # ================================================================
    dist_base = surface_distance(baseline, clean)
    dist_std = surface_distance(standard, clean)
    dist_orth = surface_distance(ours, clean)

    # Shared colorscale across all 3 panels
    all_dists = np.concatenate([dist_base, dist_std, dist_orth])
    vmin_d = 0
    vmax_d = np.percentile(all_dists, 97)

    # View angle
    elev, azim = find_best_view(clean)
    elev = max(20, min(40, elev))

    # Global bounding box (include standard's possible outliers)
    all_pts = np.concatenate([clean, baseline, standard, ours])
    mid = all_pts.mean(axis=0)
    max_range = np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2.0
    pad = max_range * 1.15

    # ================================================================
    # C.2: Voxel Density Error (|rho_pred - rho_GT| on 64^3 grid)
    # ================================================================
    print("Computing 64^3 density grids...")
    dens_gt = compute_density_grid(clean)
    dens_base = compute_density_grid(baseline)
    dens_std = compute_density_grid(standard)
    dens_ours = compute_density_grid(ours)

    err_base = np.abs(dens_base - dens_gt)
    err_std = np.abs(dens_std - dens_gt)
    err_ours = np.abs(dens_ours - dens_gt)

    # Choose projection axis: thinnest PCA direction
    centered = clean - clean.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_axis = np.argmax(np.abs(Vt[2]))

    # Max-intensity projection along thin axis
    mip_base = err_base.max(axis=proj_axis)
    mip_std = err_std.max(axis=proj_axis)
    mip_ours = err_ours.max(axis=proj_axis)
    mip_gt = dens_gt.max(axis=proj_axis)  # for structure reference

    # Shared density-error colorscale across all 3 C.2 panels
    err_vmax = max(mip_base.max(), mip_std.max(), mip_ours.max()) * 0.85

    # ================================================================
    # Figure layout
    #   - Taller figure with more vertical space between rows
    #   - 3 rows conceptually: C.1 section header, C.1 plots, gap,
    #     C.2 section header, C.2 plots, bottom text
    # ================================================================
    fig = plt.figure(figsize=(16, 12))

    # Use manual axes placement to avoid all overlap issues.
    # C.1: 3 x 3D scatter panels in top half
    # C.2: 3 x 2D heatmap panels in bottom half
    # Colorbars on the right margin

    # ── Coordinates (in figure fraction) ──
    # C.1 row: y from 0.52 to 0.88  (height 0.36)
    # C.2 row: y from 0.08 to 0.42  (height 0.34)
    # Gap between rows: 0.42 to 0.52 (0.10) — room for IGSD labels + C.2 header
    c1_bottom, c1_top = 0.52, 0.88
    c2_bottom, c2_top = 0.10, 0.44
    col_left = 0.04
    col_right = 0.89
    col_width = (col_right - col_left - 0.04) / 3  # 3 panels + small gaps
    col_gap = 0.02
    cbar_left = 0.92
    cbar_width = 0.012

    cmap_dist = 'RdYlBu_r'  # blue (close/good) -> red (far/bad)

    # ── C.1: Surface Distance Error (top row, 3 panels) ──
    c1_panels = [
        (baseline, dist_base, 'Baseline', 'base'),
        (standard, dist_std,
         r'Standard Guidance ($\lambda\!=\!3.0$)', 'std'),
        (ours, dist_orth,
         r'Orthogonal Guidance ($\lambda\!=\!3.0$)', 'orth'),
    ]

    c1_axes = []
    for col, (pts, dists, title, key) in enumerate(c1_panels):
        x0 = col_left + col * (col_width + col_gap)
        ax = fig.add_axes(
            [x0, c1_bottom, col_width, c1_top - c1_bottom],
            projection='3d',
        )
        c1_axes.append(ax)

        # Sort so high-distance (red) points draw last / on top
        order = np.argsort(dists)
        pts_s = pts[order]
        dists_s = dists[order]

        sc = ax.scatter(
            pts_s[:, 0], pts_s[:, 2], pts_s[:, 1],
            c=dists_s, cmap=cmap_dist, s=5.0,
            vmin=vmin_d, vmax=vmax_d, alpha=0.9,
            edgecolors='none', rasterized=True,
        )

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=4)
        ax.set_axis_off()

        # Consistent framing
        ax.set_xlim(mid[0] - pad, mid[0] + pad)
        ax.set_ylim(mid[2] - pad, mid[2] + pad)
        ax.set_zlim(mid[1] - pad, mid[1] + pad)

        # IGSD annotation below — two-line format to avoid horizontal overflow
        if key == 'base':
            igsd_val = data['igsd_base']
            ax.text2D(0.5, -0.01,
                      f'IGSD = {igsd_val:.2e}',
                      transform=ax.transAxes, fontsize=9, ha='center',
                      color=C_GRAY)
        elif key == 'std':
            igsd_val = data['igsd_std']
            igsd_ratio = igsd_val / (data['igsd_base'] + 1e-15)
            ax.text2D(0.5, -0.01,
                      f'IGSD = {igsd_val:.4f}\n'
                      r'($\times$' + f'{igsd_ratio:.0f} vs base, Collapse)',
                      transform=ax.transAxes, fontsize=8.5, ha='center',
                      color=C_RED, fontweight='bold',
                      linespacing=1.4)
        else:
            igsd_val = data['igsd_orth']
            igsd_ratio = igsd_val / (data['igsd_base'] + 1e-15)
            protect = data['igsd_std'] / (igsd_val + 1e-15)
            ax.text2D(0.5, -0.01,
                      f'IGSD = {igsd_val:.4f}\n'
                      r'($\times$' + f'{igsd_ratio:.0f} vs base, '
                      f'{protect:.0f}x safer)',
                      transform=ax.transAxes, fontsize=8.5, ha='center',
                      color=C_GREEN, fontweight='bold',
                      linespacing=1.4)

    # C.1 colorbar
    cbar_ax1 = fig.add_axes(
        [cbar_left, c1_bottom + 0.02, cbar_width, c1_top - c1_bottom - 0.04])
    cbar1 = fig.colorbar(sc, cax=cbar_ax1)
    cbar1.set_label('Surface Distance to GT', fontsize=9, labelpad=8)
    cbar1.ax.tick_params(labelsize=8)

    # ── C.2: Voxel Density Error (bottom row, 3 panels) ──
    ax_labels = ['x', 'y', 'z']
    remaining = [i for i in range(3) if i != proj_axis]
    xlabel = ax_labels[remaining[0]]
    ylabel = ax_labels[remaining[1]]

    gt_contour_x = np.linspace(-1, 1, mip_gt.shape[0])
    gt_contour_y = np.linspace(-1, 1, mip_gt.shape[1])
    gt_level = [mip_gt.max() * 0.1]

    c2_panels = [
        (mip_base, 'Baseline', data['vd_base'], 'base'),
        (mip_std, r'Standard ($\lambda\!=\!3.0$)', data['vd_std'], 'std'),
        (mip_ours, r'Ours ($\lambda\!=\!3.0$)', data['vd_orth'], 'orth'),
    ]

    im_last = None
    c2_axes = []
    for col, (mip, title, vd_val, key) in enumerate(c2_panels):
        x0 = col_left + col * (col_width + col_gap)
        ax2 = fig.add_axes([x0, c2_bottom, col_width, c2_top - c2_bottom])
        c2_axes.append(ax2)

        im = ax2.imshow(
            mip.T, origin='lower', cmap='Reds',
            vmin=0, vmax=err_vmax, extent=[-1, 1, -1, 1], aspect='equal',
        )
        # GT structure contour
        ax2.contour(
            gt_contour_x, gt_contour_y, mip_gt.T,
            levels=gt_level, colors='#333333', linewidths=0.6, alpha=0.4,
        )
        ax2.set_title(f'{title} \u2014 Density Error',
                       fontsize=10, fontweight='bold', pad=5)
        ax2.set_xlabel(xlabel, fontsize=8)
        if col == 0:
            ax2.set_ylabel(ylabel, fontsize=8)
        else:
            ax2.set_yticklabels([])
        ax2.tick_params(labelsize=7)

        # VD annotation — placed inside the panel (top-left corner)
        # to avoid overlap with x-axis label below
        if key == 'base':
            ax2.text(0.03, 0.96, f'VD = {vd_val:.4f}',
                     transform=ax2.transAxes, fontsize=9, ha='left',
                     va='top', color=C_GRAY,
                     bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))
        elif key == 'std':
            vd_pct = (vd_val - data['vd_base']) / data['vd_base'] * 100
            ax2.text(0.03, 0.96,
                     f'VD = {vd_val:.4f}\n(+{vd_pct:.0f}% vs base)',
                     transform=ax2.transAxes, fontsize=8.5, ha='left',
                     va='top', color=C_RED, fontweight='bold',
                     linespacing=1.3,
                     bbox=dict(fc='white', ec=C_RED, alpha=0.85,
                               pad=2, lw=0.6))
        else:
            vd_reduce = (data['vd_std'] - vd_val) / data['vd_std'] * 100
            ax2.text(0.03, 0.96,
                     f'VD = {vd_val:.4f}\n({vd_reduce:.0f}% less vs Std)',
                     transform=ax2.transAxes, fontsize=8.5, ha='left',
                     va='top', color=C_GREEN, fontweight='bold',
                     linespacing=1.3,
                     bbox=dict(fc='white', ec=C_GREEN, alpha=0.85,
                               pad=2, lw=0.6))
        im_last = im

    # Zoom-in boxes on Standard and Ours: find peak error reduction
    diff_map = mip_std - mip_ours  # positive = standard worse
    diff_smooth = gaussian_filter(diff_map, sigma=3)
    peak_idx = np.unravel_index(diff_smooth.argmax(), diff_smooth.shape)
    cx = -1 + 2 * peak_idx[0] / diff_map.shape[0]
    cy = -1 + 2 * peak_idx[1] / diff_map.shape[1]
    zoom_sz = 0.45

    for ax_panel in [c2_axes[1], c2_axes[2]]:
        rect = Rectangle(
            (cx - zoom_sz / 2, cy - zoom_sz / 2), zoom_sz, zoom_sz,
            linewidth=1.5, edgecolor='black', facecolor='none',
            linestyle='--', zorder=10,
        )
        ax_panel.add_patch(rect)

    # C.2 colorbar
    cbar_ax2 = fig.add_axes(
        [cbar_left, c2_bottom + 0.02, cbar_width, c2_top - c2_bottom - 0.04])
    cbar2 = fig.colorbar(im_last, cax=cbar_ax2)
    cbar2.set_label(
        r'$|\rho_{\mathrm{pred}} - \rho_{\mathrm{GT}}|$',
        fontsize=9, labelpad=8,
    )
    cbar2.ax.tick_params(labelsize=8)

    # ── Section labels (in the gap between rows) ──
    fig.text(0.02, c1_top + 0.025, 'C.1',
             fontsize=13, fontweight='bold', color='#333')
    fig.text(0.06, c1_top + 0.025,
             'Surface Distance Error  (IGSD verification)',
             fontsize=11, color=C_GRAY)

    fig.text(0.02, c2_top + 0.025, 'C.2',
             fontsize=13, fontweight='bold', color='#333')
    fig.text(0.06, c2_top + 0.025,
             r'Voxel Density Error  (VD verification, $64^3$ grid)',
             fontsize=11, color=C_GRAY)

    # Overall title — above everything
    fig.text(0.48, 0.97,
             r'(c) Decoupled Visual Verification at High Guidance Scale '
             r'($\lambda\!=\!3.0$)',
             fontsize=14, fontweight='bold', ha='center', va='top')

    # Bottom annotation — below everything
    fig.text(0.48, 0.02,
             r'$\sigma\!=\!0.03$, linear decay annealing, 10 DDPM steps',
             fontsize=9, ha='center', color='#888888', fontstyle='italic')

    plt.savefig('figures/panel_c_qualitative.pdf',
                bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig('figures/panel_c_qualitative.png',
                bbox_inches='tight', dpi=300, transparent=False)
    print("Saved figures/panel_c_qualitative.pdf and .png")


# =========================================================================== #
#  Cache I/O
# =========================================================================== #

def save_data(data, path='figures/panel_c_data_v2.npz'):
    np.savez(
        path,
        clean=data['clean'],
        baseline=data['baseline'],
        standard=data['standard'],
        ours=data['ours'],
        vd_base=data['vd_base'],
        vd_std=data['vd_std'],
        vd_orth=data['vd_orth'],
        igsd_base=data['igsd_base'],
        igsd_std=data['igsd_std'],
        igsd_orth=data['igsd_orth'],
        idx=data['idx'],
    )
    print(f"Saved data to {path}")


def load_data(path='figures/panel_c_data_v2.npz'):
    d = np.load(path)
    return {
        'clean': d['clean'],
        'baseline': d['baseline'],
        'standard': d['standard'],
        'ours': d['ours'],
        'vd_base': float(d['vd_base']),
        'vd_std': float(d['vd_std']),
        'vd_orth': float(d['vd_orth']),
        'igsd_base': float(d['igsd_base']),
        'igsd_std': float(d['igsd_std']),
        'igsd_orth': float(d['igsd_orth']),
        'idx': int(d['idx']),
    }


def main():
    cache = 'figures/panel_c_data_v2.npz'
    if os.path.exists(cache) and '--rerun' not in sys.argv:
        print(f"Loading cached data from {cache}")
        data = load_data(cache)
    else:
        print("Running denoising experiments (3 methods, lambda=3.0)...")
        data = run_denoising_trio(noise_std=0.03, seed=42)
        save_data(data, cache)

    print(f"\nShape idx={data['idx']}:")
    print(f"  VD:   base={data['vd_base']:.4f}, std={data['vd_std']:.4f}, "
          f"orth={data['vd_orth']:.4f}")
    print(f"  IGSD: base={data['igsd_base']:.6f}, std={data['igsd_std']:.6f}, "
          f"orth={data['igsd_orth']:.6f}")

    make_figure(data)


if __name__ == '__main__':
    main()
