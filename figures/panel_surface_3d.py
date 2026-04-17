"""Panel: 3D Surface Renderings — Clean surface with s_t and Noisy surface with g_t."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d  # noqa: F401 – registers 3D projection
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

# ── Colour palette (matches panel_a_mechanism.py) ──
C_SURF_CLEAN = '#4393C3'
C_WIRE_CLEAN = '#2166AC'
C_SURF_NOISY = '#92C5DE'
C_WIRE_NOISY = '#67A9CF'
C_SCORE = '#2166AC'
C_GRAD = '#B2182B'
C_NOISE_PT = '#D6604D'

# ── Camera ──
ELEV, AZIM = 25, -45


# ====================================================================
# 3‑D arrow that respects matplotlib's 3‑D projection pipeline
# ====================================================================
class Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch that lives in data‑space (x, y, z)."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs2d, ys2d, zs2d = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return min(zs2d)


# ====================================================================
# Surface helpers
# ====================================================================
def make_surface(nx_pts=40):
    """Return X, Y, Z for a Gaussian bump on [-1.5, 1.5]²."""
    u = np.linspace(-1.5, 1.5, nx_pts)
    X, Y = np.meshgrid(u, u)
    Z = 0.6 * np.exp(-(X ** 2 + Y ** 2) / 0.8)
    return X, Y, Z


def style_3d_axes(ax):
    """Minimalist 3‑D axis styling: no ticks, light pane faces."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.35))
    ax.yaxis.pane.set_facecolor((0.93, 0.93, 0.93, 0.35))
    ax.zaxis.pane.set_facecolor((0.91, 0.91, 0.91, 0.35))
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.5))
    ax.view_init(elev=ELEV, azim=AZIM)


# ====================================================================
# Panel 1 — Clean surface + s_t
# ====================================================================
def render_clean():
    X, Y, Z = make_surface()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    ax.plot_surface(X, Y, Z, color=C_SURF_CLEAN, alpha=0.85,
                    edgecolor='none', shade=True, lightsource=None)
    ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4,
                      color=C_WIRE_CLEAN, linewidth=0.4, alpha=0.55)

    # Arrow s_t — from peak, pointing upward‑forward (approximate normal)
    peak = np.array([0.0, 0.0, 0.6])
    direction = np.array([0.15, -0.15, 0.75])  # mostly upward, slight forward
    direction = direction / np.linalg.norm(direction) * 0.7
    tip = peak + direction

    arrow = Arrow3D(
        [peak[0], tip[0]], [peak[1], tip[1]], [peak[2], tip[2]],
        arrowstyle='-|>', mutation_scale=30, lw=5.0,
        color=C_SCORE, shrinkA=0, shrinkB=0,
    )
    ax.add_artist(arrow)

    # Label
    label_pos = tip + np.array([0.05, -0.05, 0.08])
    ax.text(label_pos[0], label_pos[1], label_pos[2],
            r'$\mathbf{s}_t$', fontsize=26, fontweight='bold',
            color=C_SCORE, zorder=20)

    style_3d_axes(ax)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-0.05, 1.1)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/figure_clean_surface.{ext}',
                    bbox_inches='tight', dpi=300, transparent=True)
    print("Saved figures/figure_clean_surface.pdf and .png")
    plt.close(fig)


# ====================================================================
# Panel 2 — Noisy surface + scattered points + g_t
# ====================================================================
def render_noisy():
    X, Y, Z = make_surface()
    rng = np.random.default_rng(42)

    # Perturbed surface
    Z_noisy = Z + rng.normal(0, 0.035, Z.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    ax.plot_surface(X, Y, Z_noisy, color=C_SURF_NOISY, alpha=0.75,
                    edgecolor='none', shade=True)
    ax.plot_wireframe(X, Y, Z_noisy, rstride=4, cstride=4,
                      color=C_WIRE_NOISY, linewidth=0.4, alpha=0.50)

    # Scattered noise points around the surface
    n_pts = 120
    idx = rng.integers(0, X.size, n_pts)
    sx = X.ravel()[idx] + rng.normal(0, 0.08, n_pts)
    sy = Y.ravel()[idx] + rng.normal(0, 0.08, n_pts)
    sz = Z.ravel()[idx] + rng.normal(0, 0.10, n_pts)
    ax.scatter(sx, sy, sz, c=C_NOISE_PT, s=12, alpha=0.65,
               edgecolors='none', depthshade=True, zorder=5)

    # Arrow g_t — from a point on the surface, pointing right‑forward
    origin = np.array([0.0, 0.0, 0.6])
    direction = np.array([0.65, 0.45, 0.15])  # mostly lateral
    direction = direction / np.linalg.norm(direction) * 0.7
    tip = origin + direction

    arrow = Arrow3D(
        [origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]],
        arrowstyle='-|>', mutation_scale=30, lw=5.0,
        color=C_GRAD, shrinkA=0, shrinkB=0,
    )
    ax.add_artist(arrow)

    # Label
    label_pos = tip + np.array([0.08, 0.05, 0.06])
    ax.text(label_pos[0], label_pos[1], label_pos[2],
            r'$\mathbf{g}_t$', fontsize=26, fontweight='bold',
            color=C_GRAD, zorder=20)

    style_3d_axes(ax)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-0.05, 1.1)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/figure_noisy_surface.{ext}',
                    bbox_inches='tight', dpi=300, transparent=True)
    print("Saved figures/figure_noisy_surface.pdf and .png")
    plt.close(fig)


# ====================================================================
if __name__ == '__main__':
    render_clean()
    render_noisy()
