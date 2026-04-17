"""
Contour Plot with vector decomposition + 3D Vector Field on Surface.

Panel 1 (Contour):  Top-down level-set view of the Gaussian bump.
    At a sample point x_t, shows s_t (blue), raw g_t (red dashed),
    and the tangential projection Proj (green) with a right-angle marker.

Panel 2 (Vector Field on Surface):  3D surface with quiver arrows
    showing raw gradient (red) vs. tangential projection (green)
    across the manifold — visual proof that our method keeps guidance
    tangent to the data manifold.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# ── Colour palette ──
C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_GREEN = '#1B7837'
C_GRAY = '#555555'
C_SURF = '#4393C3'
C_WIRE = '#2166AC'
C_LIGHT_RED = '#D6604D'

ELEV, AZIM = 25, -45


def _surface(x, y):
    """z = 0.6 exp(-(x²+y²)/0.8)"""
    return 0.6 * np.exp(-(x ** 2 + y ** 2) / 0.8)


def _grad_z(x, y):
    """Analytical ∂z/∂x, ∂z/∂y."""
    e = np.exp(-(x ** 2 + y ** 2) / 0.8)
    dzdx = 0.6 * e * (-2 * x / 0.8)
    dzdy = 0.6 * e * (-2 * y / 0.8)
    return dzdx, dzdy


# ====================================================================
# Arrow3D for the vector-field panel
# ====================================================================
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs2d, ys2d, zs2d = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return min(zs2d)


# ====================================================================
# Panel 1 — Contour plot with vector decomposition
# ====================================================================
def render_contour():
    grid = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(grid, grid)
    Z = _surface(X, Y)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Filled contours + line contours
    levels = np.linspace(0.0, 0.6, 13)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.55)
    cs = ax.contour(X, Y, Z, levels=levels, colors=C_GRAY, linewidths=0.6,
                    alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f', colors=C_GRAY)

    # ── Anchor point x_t (off-peak) ──
    xt = np.array([0.55, 0.40])
    ax.plot(*xt, 'o', color='#222222', ms=10, zorder=12)
    ax.text(xt[0] + 0.10, xt[1] + 0.10, r'$x_t$', fontsize=18,
            fontweight='bold', color='#222222', zorder=12)

    # ── s_t: score vector (toward peak ≈ negative gradient of potential) ──
    dzdx, dzdy = _grad_z(xt[0], xt[1])
    s_dir = np.array([dzdx, dzdy])           # points toward higher z = toward peak
    s_dir = s_dir / np.linalg.norm(s_dir)
    s_len = 0.55
    s_vec = s_dir * s_len

    ax.annotate('', xy=xt + s_vec, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=5.0,
                                mutation_scale=28), zorder=10)
    # Label: offset perpendicular to arrow so it doesn't overlap
    s_perp = np.array([-s_dir[1], s_dir[0]])  # perpendicular direction
    s_label = xt + s_vec * 0.6 + s_perp * 0.14
    ax.text(s_label[0], s_label[1], r'$\mathbf{s}_t$', fontsize=26,
            fontweight='bold', color=C_BLUE, ha='center', va='center',
            zorder=12)

    # ── g_t: raw geometric gradient (angled — has both normal + tangent) ──
    g_angle = np.arctan2(s_vec[1], s_vec[0]) + 0.95   # ~55° off s_t
    g_dir = np.array([np.cos(g_angle), np.sin(g_angle)])
    g_len = 0.65
    g_vec = g_dir * g_len

    ax.annotate('', xy=xt + g_vec, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=4.0,
                                ls='--', mutation_scale=24), zorder=9)
    # Label with box background for readability
    g_label = xt + g_vec + g_dir * 0.06 + np.array([0.08, -0.10])
    ax.text(g_label[0], g_label[1], r'$\nabla L_{\mathrm{geom}}$',
            fontsize=20, fontweight='bold', color=C_RED,
            ha='left', va='top', zorder=12,
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='none', alpha=0.80))

    # ── Decomposition ──
    # Normal component (along s_t)
    g_dot_s = np.dot(g_vec, s_dir)
    gn_vec = g_dot_s * s_dir      # normal part (to be removed)
    gt_vec = g_vec - gn_vec        # tangential part (kept)

    # Harmful normal component (thin red dotted)
    ax.annotate('', xy=xt + gn_vec, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5,
                                ls=':', alpha=0.45, mutation_scale=13),
                zorder=7)

    # X mark on normal component
    xm = xt + gn_vec * 0.55
    ax.plot(xm[0], xm[1], 'x', color=C_RED, ms=20, mew=4.0, zorder=13)

    # Tangential projection (GREEN — the useful component)
    ax.annotate('', xy=xt + gt_vec, xytext=xt,
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=5.0,
                                mutation_scale=28), zorder=10)
    gt_hat = gt_vec / (np.linalg.norm(gt_vec) + 1e-12)
    gt_perp = np.array([-gt_hat[1], gt_hat[0]])
    gt_label = xt + gt_vec + gt_hat * 0.05 + gt_perp * 0.13
    ax.text(gt_label[0], gt_label[1],
            r'$\mathrm{Proj}_{\perp \mathbf{s}}$',
            fontsize=18, fontweight='bold', color=C_GREEN,
            ha='center', va='center', zorder=12,
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='none', alpha=0.80))

    # Parallelogram construction lines
    ax.plot([xt[0] + g_vec[0], xt[0] + gt_vec[0]],
            [xt[1] + g_vec[1], xt[1] + gt_vec[1]],
            color='#BBBBBB', lw=1.0, ls='--', zorder=6)
    ax.plot([xt[0] + g_vec[0], xt[0] + gn_vec[0]],
            [xt[1] + g_vec[1], xt[1] + gn_vec[1]],
            color='#BBBBBB', lw=1.0, ls='--', zorder=6)

    # Right-angle marker between s_t and Proj
    q = 0.06
    t_hat = gt_vec / (np.linalg.norm(gt_vec) + 1e-12)
    ax.plot([xt[0] + q * t_hat[0],
             xt[0] + q * t_hat[0] + q * s_dir[0],
             xt[0] + q * s_dir[0]],
            [xt[1] + q * t_hat[1],
             xt[1] + q * t_hat[1] + q * s_dir[1],
             xt[1] + q * s_dir[1]],
            color=C_GRAY, lw=1.2, zorder=8)

    # ── Legend ──
    handles = [
        Line2D([0], [0], color=C_BLUE, lw=4,
               label=r'$\mathbf{s}_t$: Score (manifold normal)'),
        Line2D([0], [0], color=C_RED, lw=3, ls='--',
               label=r'$\nabla L_{\mathrm{geom}}$: Raw gradient'),
        Line2D([0], [0], color=C_GREEN, lw=4,
               label=r'$\mathrm{Proj}_{\perp \mathbf{s}}$: '
                     r'Tangential (ours)'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=10,
              framealpha=0.93, edgecolor='#CCCCCC', fancybox=True,
              borderpad=0.6, handlelength=2.2)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Contour View: Orthogonal Decomposition at $x_t$',
                 fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/figure_contour_vectors.{ext}',
                    bbox_inches='tight', dpi=300, transparent=True)
    print("Saved figures/figure_contour_vectors.pdf and .png")
    plt.close(fig)


# ====================================================================
# Panel 2 — 3D vector field on surface (raw vs. projected)
# ====================================================================
def render_vectorfield():
    # Fine mesh for surface rendering
    u_fine = np.linspace(-1.5, 1.5, 40)
    Xf, Yf = np.meshgrid(u_fine, u_fine)
    Zf = _surface(Xf, Yf)

    # Coarse grid for quiver arrows
    u_q = np.linspace(-1.1, 1.1, 8)
    Xq, Yq = np.meshgrid(u_q, u_q)
    Zq = _surface(Xq, Yq)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    ax.plot_surface(Xf, Yf, Zf, color=C_SURF, alpha=0.60,
                    edgecolor='none', shade=True)
    ax.plot_wireframe(Xf, Yf, Zf, rstride=4, cstride=4,
                      color=C_WIRE, linewidth=0.3, alpha=0.35)

    # ── Define a "geometric loss" target direction (global push toward +x) ──
    # This simulates a geometric guidance gradient pointing roughly toward
    # a target shape (rightward in world coords).
    target_dir_xy = np.array([0.8, 0.3])
    target_dir_xy = target_dir_xy / np.linalg.norm(target_dir_xy)

    scale = 0.18  # arrow visual length

    for i in range(Xq.shape[0]):
        for j in range(Xq.shape[1]):
            x0, y0, z0 = Xq[i, j], Yq[i, j], Zq[i, j]

            # Skip corners (too far from bump — arrows look odd)
            if x0 ** 2 + y0 ** 2 > 1.35 ** 2:
                continue

            # Surface normal at (x0, y0)
            dzdx, dzdy = _grad_z(x0, y0)
            n = np.array([-dzdx, -dzdy, 1.0])
            n = n / np.linalg.norm(n)

            # Raw gradient in 3D: target_dir lifted to surface slope
            # gz component comes from chain rule along surface
            gz = dzdx * target_dir_xy[0] + dzdy * target_dir_xy[1]
            g_raw = np.array([target_dir_xy[0], target_dir_xy[1], gz])
            g_raw = g_raw / (np.linalg.norm(g_raw) + 1e-12) * scale

            # Tangential projection: remove normal component
            g_proj = g_raw - np.dot(g_raw, n) * n
            g_proj = g_proj / (np.linalg.norm(g_proj) + 1e-12) * scale

            # Raw gradient arrow (red)
            ax.quiver(x0, y0, z0,
                      g_raw[0], g_raw[1], g_raw[2],
                      color=C_RED, alpha=0.70, linewidth=1.6,
                      arrow_length_ratio=0.25)

            # Projected tangential arrow (green)
            ax.quiver(x0, y0, z0,
                      g_proj[0], g_proj[1], g_proj[2],
                      color=C_GREEN, alpha=0.85, linewidth=1.8,
                      arrow_length_ratio=0.25)

    # ── Highlight one point with labelled thick arrows ──
    xh, yh = 0.4, 0.3
    zh = _surface(xh, yh)
    dzdx_h, dzdy_h = _grad_z(xh, yh)
    n_h = np.array([-dzdx_h, -dzdy_h, 1.0])
    n_h = n_h / np.linalg.norm(n_h)

    gz_h = dzdx_h * target_dir_xy[0] + dzdy_h * target_dir_xy[1]
    g_raw_h = np.array([target_dir_xy[0], target_dir_xy[1], gz_h])
    g_raw_h = g_raw_h / np.linalg.norm(g_raw_h) * 0.45

    g_proj_h = g_raw_h - np.dot(g_raw_h, n_h) * n_h
    g_proj_h = g_proj_h / np.linalg.norm(g_proj_h) * 0.45

    # Score / normal arrow (blue, upward)
    s_h = n_h * 0.45
    a_s = Arrow3D([xh, xh + s_h[0]], [yh, yh + s_h[1]],
                  [zh, zh + s_h[2]],
                  arrowstyle='-|>', mutation_scale=26, lw=4.5,
                  color=C_BLUE, shrinkA=0, shrinkB=0)
    ax.add_artist(a_s)
    ax.text(xh + s_h[0] + 0.02, yh + s_h[1] - 0.05, zh + s_h[2] + 0.06,
            r'$\mathbf{s}_t$', fontsize=24, fontweight='bold',
            color=C_BLUE, zorder=20)

    # Raw gradient (red)
    a_g = Arrow3D([xh, xh + g_raw_h[0]], [yh, yh + g_raw_h[1]],
                  [zh, zh + g_raw_h[2]],
                  arrowstyle='-|>', mutation_scale=22, lw=3.5,
                  color=C_RED, linestyle='--', shrinkA=0, shrinkB=0)
    ax.add_artist(a_g)
    ax.text(xh + g_raw_h[0] + 0.12, yh + g_raw_h[1] + 0.15,
            zh + g_raw_h[2] + 0.10,
            r'$\nabla L$', fontsize=22, fontweight='bold',
            color=C_RED, zorder=20)

    # Projected (green)
    a_p = Arrow3D([xh, xh + g_proj_h[0]], [yh, yh + g_proj_h[1]],
                  [zh, zh + g_proj_h[2]],
                  arrowstyle='-|>', mutation_scale=26, lw=4.5,
                  color=C_GREEN, shrinkA=0, shrinkB=0)
    ax.add_artist(a_p)
    ax.text(xh + g_proj_h[0] + 0.10, yh + g_proj_h[1] - 0.20,
            zh + g_proj_h[2] - 0.10,
            r'$\mathrm{Proj}_{\perp \mathbf{s}}$', fontsize=20,
            fontweight='bold', color=C_GREEN, zorder=20)

    # ── Legend (manual, upper-left via proxy artists) ──
    legend_elements = [
        Line2D([0], [0], color=C_BLUE, lw=4,
               label=r'$\mathbf{s}_t$ (score / normal)'),
        Line2D([0], [0], color=C_RED, lw=3, ls='--',
               label=r'$\nabla L$ (raw gradient)'),
        Line2D([0], [0], color=C_GREEN, lw=4,
               label=r'$\mathrm{Proj}_{\perp \mathbf{s}}$ (tangential)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.90, edgecolor='#CCCCCC', fancybox=True)

    # ── Axes styling ──
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.35))
    ax.yaxis.pane.set_facecolor((0.93, 0.93, 0.93, 0.35))
    ax.zaxis.pane.set_facecolor((0.91, 0.91, 0.91, 0.35))
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-0.05, 1.0)
    ax.set_title('Vector Field: Raw Gradient vs. Tangential Projection',
                 fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/figure_vectorfield_surface.{ext}',
                    bbox_inches='tight', dpi=300, transparent=True)
    print("Saved figures/figure_vectorfield_surface.pdf and .png")
    plt.close(fig)


# ====================================================================
if __name__ == '__main__':
    render_contour()
    render_vectorfield()
