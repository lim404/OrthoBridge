"""Panel B: Guidance Trajectories and Manifold Preservation.

Relative-% scatter plot showing VD% vs IGSD% degradation for:
  - Baseline at origin (0, 0)
  - Standard Guidance trajectory (red) — collapses at high λ
  - Orthogonal Guidance trajectory (green) — controlled degradation

Uses pooled data from 3 data realizations × 40 shapes = 120 samples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from scipy import stats
from omegaconf import OmegaConf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_GREEN = '#1B7837'
C_GRAY = '#666666'


def load_clean_patches(cfg, device, num_shapes=40):
    from sb_cover.data.punet_loader import get_punet_loaders
    from models.train_utils import to_cuda
    # Fix all seeds for fully deterministic patch extraction
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
    for batch in test_loader:
        if n >= num_shapes: break
        batch = to_cuda(batch, device)
        x = batch["clean_points"].squeeze()
        if x.dim() == 2: x = x.unsqueeze(0)
        if x.shape[1] > x.shape[2]: x = x.transpose(1, 2)
        use = min(x.shape[0], num_shapes - n)
        all_clean.append(x[:use]); n += use
    return torch.cat(all_clean)


def compute_metrics(x_pred, x_clean):
    from metrics.geometric_metrics import compute_vd, compute_igsd
    B = x_pred.shape[0]
    vds, igsds = [], []
    for i in range(B):
        p = x_pred[i].transpose(0, 1)
        c = x_clean[i].transpose(0, 1)
        vds.append(compute_vd(p, c))
        igsds.append(compute_igsd(p, c))
    return np.array(vds), np.array(igsds)


def run_experiments():
    """Run all configurations across multiple data realizations for robust stats."""
    from sb_cover.training.trainer_igv import TrainerIGV
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss,
    )

    cfg = OmegaConf.load("configs/shapenet_denoise_sb_igv.yaml")
    device = torch.device("cuda")

    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    gl = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    noise_std = 0.03
    num_shapes = 40

    # Labels for all configs
    std_lambdas = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    orth_lambdas = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

    all_labels = ["Baseline"]
    all_labels += [f"Std λ={l}" for l in std_lambdas]
    all_labels += [f"Orth λ={l}" for l in orth_lambdas]

    # Accumulate per-shape metrics across realizations
    accum = {lab: ([], []) for lab in all_labels}

    for trial, noise_seed in enumerate([42, 123, 777]):
        print(f"\n=== Trial {trial+1}/3 (noise_seed={noise_seed}) ===")

        # Load patches (different random extraction each trial)
        torch.manual_seed(trial * 1000)
        np.random.seed(trial * 1000)
        x_clean = load_clean_patches(cfg, device, num_shapes=num_shapes)
        N = x_clean.shape[0]
        print(f"  Loaded {N} shapes")

        rng = torch.Generator(device=device)
        rng.manual_seed(noise_seed)
        x_noisy = x_clean + noise_std * torch.randn(
            x_clean.shape, device=device, generator=rng)

        # Baseline
        print("  Baseline...", end="", flush=True)
        with torch.no_grad():
            r = ddpm_denoise(trainer.model, trainer.sb_schedule, x_noisy,
                             sampling_steps=10, verbose=False)
        vd, igsd = compute_metrics(r["x_pred"], x_clean)
        accum["Baseline"][0].append(vd)
        accum["Baseline"][1].append(igsd)
        print(f" VD={vd.mean():.4f} IGSD={igsd.mean():.6f}")

        # Standard guidance
        for lam in std_lambdas:
            label = f"Std λ={lam}"
            print(f"  {label}...", end="", flush=True)
            r = ddpm_denoise_guided(
                trainer.model, trainer.sb_schedule, x_noisy,
                sampling_steps=10, guidance_scale=lam,
                annealing="linear_decay", geom_loss=gl,
                grad_clip=2.0, verbose=False)
            vd, igsd = compute_metrics(r["x_pred"], x_clean)
            accum[label][0].append(vd)
            accum[label][1].append(igsd)
            print(f" VD={vd.mean():.4f} IGSD={igsd.mean():.6f}")

        # Orthogonal guidance
        for lam in orth_lambdas:
            label = f"Orth λ={lam}"
            print(f"  {label}...", end="", flush=True)
            r = ddpm_denoise_ortho_guided(
                trainer.model, trainer.sb_schedule, x_noisy,
                sampling_steps=10, guidance_scale=lam,
                annealing="linear_decay", geom_loss=gl,
                grad_clip=2.0, verbose=False)
            vd, igsd = compute_metrics(r["x_pred"], x_clean)
            accum[label][0].append(vd)
            accum[label][1].append(igsd)
            print(f" VD={vd.mean():.4f} IGSD={igsd.mean():.6f}")

    # Pool across trials
    results = {}
    for lab in all_labels:
        vd_pooled = np.concatenate(accum[lab][0])
        igsd_pooled = np.concatenate(accum[lab][1])
        results[lab] = (vd_pooled, igsd_pooled)
        print(f"  {lab}: VD={vd_pooled.mean():.4f}±{vd_pooled.std():.4f}  "
              f"IGSD={igsd_pooled.mean():.6f}±{igsd_pooled.std():.6f}  (N={len(vd_pooled)})")

    return results


def make_figure(results):
    """Relative-% trajectory plot: Baseline at origin, axes = degradation %."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.2))

    vd0, igsd0 = results["Baseline"]
    vd0_m, igsd0_m = vd0.mean(), igsd0.mean()

    # ── Convert all data to relative % ──
    def to_pct(vd_arr, igsd_arr):
        return ((vd_arr.mean() - vd0_m) / vd0_m * 100,
                (igsd_arr.mean() - igsd0_m) / igsd0_m * 100,
                vd_arr.std() / np.sqrt(len(vd_arr)) / vd0_m * 100,
                igsd_arr.std() / np.sqrt(len(igsd_arr)) / igsd0_m * 100)

    std_data, orth_data = {}, {}
    for label, (vd, igsd) in results.items():
        if label.startswith("Std"):
            lam = label.split("=")[1]
            std_data[lam] = to_pct(vd, igsd)
        elif label.startswith("Orth"):
            lam = label.split("=")[1]
            orth_data[lam] = to_pct(vd, igsd)

    # Clip: exclude points > 400% degradation (λ=5.0 outlier)
    clip_pct = 400
    std_lams = sorted([l for l, (vp, ip, _, _) in std_data.items()
                       if vp < clip_pct and ip < clip_pct], key=float)
    std_out_lams = sorted([l for l, (vp, ip, _, _) in std_data.items()
                           if vp >= clip_pct or ip >= clip_pct], key=float)
    orth_lams = sorted([l for l, (vp, ip, _, _) in orth_data.items()
                        if vp < clip_pct and ip < clip_pct], key=float)

    # ── Axis limits ──
    all_igsd_pct = [0] + [std_data[l][1] for l in std_lams] + \
                   [orth_data[l][1] for l in orth_lams]
    all_vd_pct = [0] + [std_data[l][0] for l in std_lams] + \
                 [orth_data[l][0] for l in orth_lams]
    x_lo = min(all_igsd_pct) - 18
    x_hi = max(all_igsd_pct) + 35
    y_lo = min(all_vd_pct) - 30
    y_hi = max(all_vd_pct) + 25

    # ── Quadrant shading (origin = baseline) ──
    ax.fill_between([x_lo, 0], y_lo, 0,
                     color=C_GREEN, alpha=0.08, zorder=0)
    ax.fill_between([0, x_hi], 0, y_hi,
                     color=C_RED, alpha=0.06, zorder=0)

    # Quadrant labels
    ax.text(x_lo * 0.50, y_lo * 0.55,
            'Both Better', fontsize=13, color=C_GREEN,
            alpha=0.50, ha='center', va='center', fontstyle='italic',
            fontweight='bold')
    ax.text(x_hi * 0.50, y_hi * 0.55,
            'Both Worse', fontsize=13, color=C_RED,
            alpha=0.40, ha='center', va='center', fontstyle='italic',
            fontweight='bold')

    # Crosshairs at origin
    ax.axhline(y=0, color=C_GRAY, lw=1.2, ls='--', alpha=0.5, zorder=2)
    ax.axvline(x=0, color=C_GRAY, lw=1.2, ls='--', alpha=0.5, zorder=2)

    # ── Baseline marker at origin ──
    ax.plot(0, 0, 's', color=C_GRAY, markersize=12, markeredgecolor='black',
            markeredgewidth=1.2, zorder=11)
    ax.annotate('Baseline (P2P-Bridge)', xy=(0, 0),
                xytext=(50, -20),
                fontsize=9.5, color=C_GRAY, fontweight='bold', ha='left',
                va='top',
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=0.9,
                                alpha=0.5))

    # ══════════════════════════════════════════
    # TRAJECTORY 1: Standard Guidance (RED)
    # ══════════════════════════════════════════
    s_x = [std_data[l][1] for l in std_lams]
    s_y = [std_data[l][0] for l in std_lams]
    ax.plot(s_x, s_y, color=C_RED, lw=2.0, ls='-', alpha=0.40, zorder=5)

    for l in std_lams:
        vp, ip, vse, ise = std_data[l]
        ax.errorbar(ip, vp, xerr=ise, yerr=vse,
                    fmt='o', color=C_RED, markersize=7.5,
                    markeredgecolor='darkred', markeredgewidth=0.8,
                    capsize=2.5, capthick=0.8, elinewidth=0.8,
                    alpha=0.85, zorder=8)

    # ══════════════════════════════════════════
    # TRAJECTORY 2: Orthogonal Guidance (GREEN)
    # ══════════════════════════════════════════
    o_x = [orth_data[l][1] for l in orth_lams]
    o_y = [orth_data[l][0] for l in orth_lams]
    ax.plot(o_x, o_y, color=C_GREEN, lw=2.0, ls='-', alpha=0.40, zorder=5)

    for l in orth_lams:
        vp, ip, vse, ise = orth_data[l]
        ax.errorbar(ip, vp, xerr=ise, yerr=vse,
                    fmt='*', color=C_GREEN, markersize=15,
                    markeredgecolor='darkgreen', markeredgewidth=1.0,
                    capsize=2.5, capthick=0.8, elinewidth=0.8,
                    zorder=9)

    # ══════════════════════════════════════════
    # BRIDGING ARROWS at λ=2.0 and λ=3.0
    # ══════════════════════════════════════════
    for l in ['2.0', '3.0']:
        if l in std_data and l in orth_data:
            svp, sip = std_data[l][0], std_data[l][1]
            ovp, oip = orth_data[l][0], orth_data[l][1]
            lw = 1.8 if l == '3.0' else 1.2
            ax.annotate('', xy=(oip, ovp), xytext=(sip, svp),
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                        lw=lw, alpha=0.50,
                                        connectionstyle='arc3,rad=0.08'),
                        zorder=6)

    # ══════════════════════════════════════════
    # λ LABELS
    # ══════════════════════════════════════════
    # Standard: label selected λ values — right of point
    std_label_set = {'0.5': 'above', '1.0': 'right', '2.0': 'above', '3.0': 'above'}
    for l, pos in std_label_set.items():
        if l not in std_data or l not in std_lams:
            continue
        vp, ip, vse, ise = std_data[l]
        if pos == 'above':
            ax.annotate(f'$\\lambda$={l}', xy=(ip, vp),
                        xytext=(ip, vp + max(vse, 3) + 6),
                        fontsize=8, color=C_RED, ha='center', alpha=0.85)
        else:
            ax.annotate(f'$\\lambda$={l}', xy=(ip, vp),
                        xytext=(ip + max(ise, 3) + 5, vp),
                        fontsize=8, color=C_RED, ha='left', va='center',
                        alpha=0.85)

    # Orthogonal: label fewer to reduce clutter — below or left
    orth_label_set = {'1.0': 'below', '2.0': 'below', '3.0': 'left'}
    for l, pos in orth_label_set.items():
        if l not in orth_data or l not in orth_lams:
            continue
        vp, ip, vse, ise = orth_data[l]
        fw = 'bold'
        if pos == 'below':
            ax.annotate(f'$\\lambda$={l}', xy=(ip, vp),
                        xytext=(ip, vp - max(vse, 3) - 7),
                        fontsize=8, color=C_GREEN, ha='center',
                        fontweight=fw, alpha=0.85)
        else:
            ax.annotate(f'$\\lambda$={l}', xy=(ip, vp),
                        xytext=(ip - max(ise, 3) - 5, vp),
                        fontsize=8, color=C_GREEN, ha='right', va='center',
                        fontweight=fw, alpha=0.85)

    # ══════════════════════════════════════════
    # TEXT CALLOUTS
    # ══════════════════════════════════════════
    # "~4× less IGSD degradation" near λ=3.0 bridging arrow
    if '3.0' in std_data and '3.0' in orth_data:
        svp3, sip3 = std_data['3.0'][0], std_data['3.0'][1]
        ovp3, oip3 = orth_data['3.0'][0], orth_data['3.0'][1]
        mid_x = (sip3 + oip3) / 2
        mid_y = (svp3 + ovp3) / 2
        ratio = sip3 / oip3 if oip3 > 0 else 0
        ax.annotate(f'~{ratio:.0f}$\\times$ less\nIGSD degradation',
                    xy=(mid_x, mid_y),
                    xytext=(mid_x + 70, mid_y - 15),
                    fontsize=9.5, color='#333333', fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', fc='#FFFDE7',
                              ec='#888888', lw=0.8, alpha=0.95),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                    lw=0.8, alpha=0.5))

    # "Low λ: marginal differences" near origin cluster
    ax.text(x_lo + 5, y_lo + 5,
            'Low $\\lambda$: marginal differences',
            fontsize=8.5, color=C_GRAY, fontstyle='italic', ha='left',
            va='bottom', alpha=0.55)

    # Arrows for clipped outliers
    for i, l in enumerate(std_out_lams):
        y_off = y_hi - 10 - 14 * i
        ax.annotate(f'Std $\\lambda$={l}  $\\rightarrow$',
                    xy=(x_hi - 5, y_off),
                    fontsize=8, color=C_RED, ha='right', alpha=0.65,
                    fontstyle='italic')

    # ── Axes ──
    ax.set_xlabel(r'$\Delta$IGSD (%)  $\longrightarrow$ Worse',
                  fontsize=13, labelpad=8)
    ax.set_ylabel(r'$\Delta$VD (%)  $\longrightarrow$ Worse',
                  fontsize=13, labelpad=8)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # ── Legend ──
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_GRAY,
               markeredgecolor='black', markersize=10,
               label='Baseline (P2P-Bridge)'),
        Line2D([0], [0], marker='o', color=C_RED, markerfacecolor=C_RED,
               markeredgecolor='darkred', markersize=7.5, linewidth=2.0,
               alpha=0.7, label='Standard Guidance'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=C_GREEN,
               markeredgecolor='darkgreen', markersize=15, linewidth=2.0,
               label='Orthogonal Guidance (Ours)'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=9.5,
              framealpha=0.93, edgecolor='#CCCCCC', fancybox=True,
              borderpad=0.8)

    ax.set_title(
        '(b) Guidance Trajectories & Manifold Preservation'
        r' ($\sigma\!=\!0.03$, $N\!=\!120$)',
        fontsize=12.5, fontweight='bold', pad=14)

    ax.grid(True, alpha=0.12, linewidth=0.5)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig('figures/panel_b_pareto.pdf', bbox_inches='tight',
                dpi=300, transparent=True)
    plt.savefig('figures/panel_b_pareto.png', bbox_inches='tight',
                dpi=300, transparent=False)
    print("Saved figures/panel_b_pareto.pdf and .png")


def save_results(results, path='figures/panel_b_data.npz'):
    """Save pooled results for fast re-plotting."""
    arrays = {}
    for label, (vd, igsd) in results.items():
        safe = label.replace(" ", "_").replace("=", "_").replace("λ", "lam")
        arrays[f'{safe}_vd'] = vd
        arrays[f'{safe}_igsd'] = igsd
    np.savez(path, labels=list(results.keys()), **arrays)
    print(f"Saved results to {path}")


def load_results(path='figures/panel_b_data.npz'):
    """Load saved results."""
    data = np.load(path, allow_pickle=True)
    labels = list(data['labels'])
    results = {}
    for label in labels:
        safe = label.replace(" ", "_").replace("=", "_").replace("λ", "lam")
        results[label] = (data[f'{safe}_vd'], data[f'{safe}_igsd'])
    return results


def main():
    import os
    cache = 'figures/panel_b_data.npz'
    if os.path.exists(cache) and '--rerun' not in os.sys.argv:
        print(f"Loading cached results from {cache}")
        results = load_results(cache)
    else:
        results = run_experiments()
        save_results(results, cache)
    make_figure(results)


if __name__ == '__main__':
    main()
