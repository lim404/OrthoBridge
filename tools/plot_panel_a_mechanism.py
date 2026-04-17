"""Panel A: Conceptual Mechanism of Orthogonal Geometric Guidance."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

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
C_GRAY = '#555555'


def M(t):
    """Manifold curve."""
    return t, 0.25 * np.sin(1.2 * t) + 0.08 * np.cos(2.5 * t)


def M_tangent(t):
    dx, dy = 1.0, 0.25*1.2*np.cos(1.2*t) - 0.08*2.5*np.sin(2.5*t)
    n = np.sqrt(dx**2 + dy**2)
    return dx/n, dy/n


def M_normal(t):
    tx, ty = M_tangent(t)
    return -ty, tx


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 7.0))

    # ── Manifold ──
    ts = np.linspace(-1.0, 6.5, 800)
    mx, my = M(ts)
    ax.fill_between(mx, my - 0.06, my + 0.06, color='#EFEFEF', alpha=0.6, zorder=1)
    ax.plot(mx, my, color='#444444', linewidth=3.0, zorder=2, solid_capstyle='round')

    # ── Anchor geometry ──
    t0 = 2.2
    px, py = M(t0)
    nx, ny = M_normal(t0)
    tx, ty = M_tangent(t0)

    # x_t high above manifold
    d_off = 1.15
    xt = (px + d_off*nx, py + d_off*ny)

    # Reference line x_t → manifold
    ax.plot([xt[0], px], [xt[1], py], color='#DDDDDD', lw=0.7, ls=':', zorder=2)
    ax.plot(px, py, 'o', color=C_GRAY, ms=4, alpha=0.4, zorder=5)

    # ── x_t marker ──
    ax.plot(*xt, 'o', color='#222', ms=10, zorder=12)
    ax.text(xt[0]-0.35, xt[1]+0.12, r'$x_t$', fontsize=17,
            fontweight='bold', color='#222', zorder=12)

    # ════════════════════════════════════════
    # SCORE vector s (BLUE, toward manifold)
    # ════════════════════════════════════════
    sL = 0.65
    sd = (-nx*sL, -ny*sL)
    s_tip = (xt[0]+sd[0], xt[1]+sd[1])
    ax.annotate('', xy=s_tip, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=3.2,
                                mutation_scale=22), zorder=8)
    # Label: far left via leader line
    ax.annotate(r'$\mathbf{s}$' + '\n(Denoising Score\n= Normal Force)',
                xy=(s_tip[0]-0.1, s_tip[1]),
                xytext=(xt[0]-2.0, s_tip[1]-0.1),
                fontsize=10, color=C_BLUE, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                          ec=C_BLUE, alpha=0.92, lw=0.8),
                arrowprops=dict(arrowstyle='-', color=C_BLUE, lw=0.6,
                                alpha=0.4, connectionstyle='arc3,rad=0.15'))

    # ════════════════════════════════════════
    # RAW gradient (RED dashed, angled up+right)
    # ════════════════════════════════════════
    gT, gN = 1.0, 0.50  # tangential / normal magnitudes
    gd = (tx*gT + nx*gN, ty*gT + ny*gN)
    g_tip = (xt[0]+gd[0], xt[1]+gd[1])
    ax.annotate('', xy=g_tip, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.3,
                                ls='--', mutation_scale=17), zorder=7)
    # Label: upper right, well clear
    ax.annotate(r'$\nabla L_{\mathrm{geom}}$' + '\n(Raw Geometric\nGradient)',
                xy=g_tip,
                xytext=(g_tip[0]+0.6, g_tip[1]+0.45),
                fontsize=10, color=C_RED, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                          ec=C_RED, alpha=0.92, lw=0.8),
                arrowprops=dict(arrowstyle='-', color=C_RED, lw=0.6,
                                alpha=0.4))

    # ════════════════════════════════════════
    # DECOMPOSITION
    # ════════════════════════════════════════
    g_dot_n = gd[0]*nx + gd[1]*ny
    gn = (g_dot_n*nx, g_dot_n*ny)       # normal component (harmful)
    gt = (gd[0]-gn[0], gd[1]-gn[1])     # tangential component (useful)
    gn_tip = (xt[0]+gn[0], xt[1]+gn[1])
    gt_tip = (xt[0]+gt[0], xt[1]+gt[1])

    # Harmful normal component (thin red dotted)
    ax.annotate('', xy=gn_tip, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2,
                                ls=':', alpha=0.45, mutation_scale=11), zorder=6)

    # X mark
    xm = (xt[0]+gn[0]*0.55, xt[1]+gn[1]*0.55)
    ax.plot(*xm, 'x', color=C_RED, ms=18, mew=3.5, zorder=11)
    # "Remove" label near X
    ax.annotate('Remove\nnormal\ncomponent',
                xy=xm, xytext=(xm[0]+0.5, xm[1]+0.35),
                fontsize=8, color=C_RED, fontstyle='italic', ha='center',
                arrowprops=dict(arrowstyle='-', color=C_RED, lw=0.5, alpha=0.35))

    # Parallelogram construction lines
    ax.plot([g_tip[0], gt_tip[0]], [g_tip[1], gt_tip[1]],
            color='#BBBBBB', lw=0.9, ls='--', zorder=5)
    ax.plot([g_tip[0], gn_tip[0]], [g_tip[1], gn_tip[1]],
            color='#BBBBBB', lw=0.9, ls='--', zorder=5)

    # Right-angle marker
    q = 0.06
    ax.plot([xt[0]+q*tx, xt[0]+q*tx+q*nx, xt[0]+q*nx],
            [xt[1]+q*ty, xt[1]+q*ty+q*ny, xt[1]+q*ny],
            color=C_GRAY, lw=1.0, zorder=6)

    # ════════════════════════════════════════
    # TANGENTIAL projection (GREEN)
    # ════════════════════════════════════════
    ax.annotate('', xy=gt_tip, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=3.2,
                                mutation_scale=22), zorder=8)
    # Label: below the green arrow tip
    ax.annotate(r'$\mathrm{Proj}_{\perp \mathbf{s}}\!(\nabla L)$'
                + '\n(Tangential Force)',
                xy=gt_tip,
                xytext=(gt_tip[0]+0.6, gt_tip[1]-0.45),
                fontsize=10, color=C_GREEN, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                          ec=C_GREEN, alpha=0.92, lw=0.8),
                arrowprops=dict(arrowstyle='-', color=C_GREEN, lw=0.6,
                                alpha=0.4))

    # ════════════════════════════════════════
    # TRAJECTORY 1: Standard (RED → off-manifold)
    # ════════════════════════════════════════
    k = 0.50
    std_end = (xt[0]+sd[0]+gd[0]*k, xt[1]+sd[1]+gd[1]*k)

    tt = np.linspace(0, 1, 80)
    mid = (xt[0]+0.5*(sd[0]+gd[0]*k)+0.07*nx,
           xt[1]+0.5*(sd[1]+gd[1]*k)+0.07*ny)
    cx = (1-tt)**2*xt[0] + 2*(1-tt)*tt*mid[0] + tt**2*std_end[0]
    cy = (1-tt)**2*xt[1] + 2*(1-tt)*tt*mid[1] + tt**2*std_end[1]
    ax.plot(cx, cy, color=C_RED, lw=1.8, ls='--', alpha=0.6, zorder=4)
    ax.plot(*std_end, 'o', color=C_RED, ms=8, zorder=10)
    ax.annotate(r'$\hat{x}_{t\text{-}1}^{\;\mathrm{std}}$',
                xy=std_end,
                xytext=(std_end[0]+0.25, std_end[1]+0.20),
                fontsize=12, color=C_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=C_RED, lw=0.5, alpha=0.4))

    # Distance to manifold
    d2 = (mx-std_end[0])**2 + (my-std_end[1])**2
    tc = ts[np.argmin(d2)]
    mc = M(tc)
    ax.plot([std_end[0], mc[0]], [std_end[1], mc[1]],
            color=C_RED, lw=1.0, ls=':', alpha=0.5, zorder=3)
    gp = ((std_end[0]+mc[0])/2+0.25, (std_end[1]+mc[1])/2)
    ax.annotate('IGSD worsens\n(pushed off $\\mathcal{M}$)',
                xy=gp, fontsize=8.5, color=C_RED,
                fontstyle='italic', ha='left', va='center')

    # ════════════════════════════════════════
    # TRAJECTORY 2: Ours (GREEN → on-manifold)
    # ════════════════════════════════════════
    ours_raw = (xt[0]+sd[0]+gt[0]*k, xt[1]+sd[1]+gt[1]*k)
    d2o = (mx-ours_raw[0])**2 + (my-ours_raw[1])**2
    to = ts[np.argmin(d2o)]
    ours_end = M(to)

    mid2 = (xt[0]+0.5*(sd[0]+gt[0]*k)-0.03*nx,
            xt[1]+0.5*(sd[1]+gt[1]*k)-0.03*ny)
    cx2 = (1-tt)**2*xt[0] + 2*(1-tt)*tt*mid2[0] + tt**2*ours_end[0]
    cy2 = (1-tt)**2*xt[1] + 2*(1-tt)*tt*mid2[1] + tt**2*ours_end[1]
    ax.plot(cx2, cy2, color=C_GREEN, lw=2.3, alpha=0.75, zorder=4)
    ax.plot(*ours_end, 'o', color=C_GREEN, ms=8, zorder=10)
    ax.annotate(r'$\hat{x}_{t\text{-}1}^{\;\mathrm{ours}}$',
                xy=ours_end,
                xytext=(ours_end[0]-0.10, ours_end[1]-0.35),
                fontsize=12, color=C_GREEN, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=C_GREEN, lw=0.5, alpha=0.4))

    # Tangential slide arrow along M
    base = M(t0)
    ax.plot(*base, 's', color=C_BLUE, ms=5, alpha=0.3, zorder=5)
    ax.annotate('', xy=(ours_end[0]-0.04, ours_end[1]-0.04),
                xytext=(base[0]+0.04, base[1]-0.04),
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.5,
                                alpha=0.4, mutation_scale=12,
                                connectionstyle='arc3,rad=-0.12'), zorder=3)
    sl = ((base[0]+ours_end[0])/2, min(base[1], ours_end[1])-0.28)
    ax.annotate('VD improves  (tangential slide on $\\mathcal{M}$)',
                xy=sl, fontsize=8.5, color=C_GREEN,
                fontstyle='italic', ha='center')

    # ── Manifold label ──
    ax.annotate(r'$\mathcal{M}$ (Data Manifold)',
                xy=(5.0, -0.38), fontsize=13, fontstyle='italic',
                color='#444444', ha='center')

    # ══════════════════
    # LEGEND
    # ══════════════════
    handles = [
        Line2D([0], [0], color=C_BLUE, lw=3.2,
               label=r'$\mathbf{s}$: Denoising score ($\approx$ manifold normal)'),
        Line2D([0], [0], color=C_RED, lw=2.3, ls='--',
               label=r'$\nabla L_{\mathrm{geom}}$: Raw gradient'
                     r' (normal $+$ tangential)'),
        Line2D([0], [0], color=C_GREEN, lw=3.2,
               label=r'$\mathrm{Proj}_{\perp \mathbf{s}}(\nabla L)$:'
                     r' Tangential guidance (ours)'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=9.5,
              framealpha=0.93, edgecolor='#CCCCCC', fancybox=True,
              borderpad=0.8, handlelength=2.5)

    # ══════════════════
    # EQUATION BOX
    # ══════════════════
    eq = (r'$\hat{x}_{t\text{-}1} = x_{t\text{-}1}^{\;\mathrm{P2P}}'
          r' \;-\; \lambda_t \cdot'
          r' \mathrm{Proj}_{\perp \mathbf{s}}'
          r'\!\left(\nabla L_{\mathrm{geom}}\right)$')
    ax.text(0.98, 0.04, eq, transform=ax.transAxes,
            fontsize=12.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='#F7F7F7',
                      ec='#888888', lw=1.0, alpha=0.95))

    # ── Axes ──
    ax.set_xlim(-1.2, 6.5)
    ax.set_ylim(-0.80, 2.55)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Orthogonal Geometric Guidance: Mechanism',
                 fontsize=15, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig('figures/panel_a_mechanism.pdf', bbox_inches='tight',
                dpi=300, transparent=True)
    plt.savefig('figures/panel_a_mechanism.png', bbox_inches='tight',
                dpi=300, transparent=False)
    print("Saved figures/panel_a_mechanism.pdf and .png")


if __name__ == '__main__':
    main()
