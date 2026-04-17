"""Premium Mitsuba 3 visualization — PUNet point cloud denoising.

Rendering standard (consistent green/red coloring by distance to GT):
  - Green = on the GT surface (clean)
  - Red   = displaced from GT surface (added noise / residual noise)
  - GT:       all green (every point is on-surface by definition)
  - Noisy:    green/red blend — the added noise shows as red
  - Denoised: green/red blend — residual noise shows as red
  - Uses ALL points (10000_poisson, no subsampling)
  - Principled BSDF (ceramic) + AO dome + 3-point directional lighting
  - Pure white background, NO text — labels go in LaTeX
  - Tight-cropped per row, minimal gap

Columns: GT | Noisy | SOR | Bilateral | P2P-Bridge | Std-0.2 | Orth-0.2
Rows:    5 PUNet test shapes

Output:
  - individual/{shape}_{method}.png   (RGBA, tight-cropped)
  - grid_overview.png / .pdf          (pure images, no text, white BG)

Usage:
    python experiments/render_punet_premium.py
    python experiments/render_punet_premium.py --noise 0.03 --spp 512
"""

import sys, os, struct, tempfile, argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

import mitsuba as mi
mi.set_variant("scalar_rgb")


# =========================================================================== #
#  Config
# =========================================================================== #
SHAPENET_DATA = "/mnt/a/Users/Administrator/PycharmProjects/ECCV/data/objects"
OUTPUT_DIR    = "experiments/figures/punet_premium"

SHAPES = [
    ("camel",    None,      (1.6, -2.2, 0.6)),
    ("horse",    None,      (1.6, -2.2, 0.6)),
    ("chair",    [0, 2, 1], (1.8, -2.0, 0.8)),
    ("elephant", [0, 2, 1], (1.8, -2.0, 0.5)),
    ("kitten",   [0, 2, 1], (1.8, -2.0, 0.5)),
]

METHODS = [
    ("P2P-Bridge", "baseline", 0.0, None),
    ("Std-0.2",    "std",      0.2, "linear_decay"),
    ("Orth-0.2",   "orth",     0.2, "linear_decay"),
]

COL_LABELS = ["GT", "Noisy", "SOR", "Bilateral", "P2P-Bridge", "Std-0.2", "Orth-0.2"]

# ── Color palette ──
VIVID_GREEN = np.array([0.10, 0.75, 0.25], dtype=np.float32)
VIVID_RED   = np.array([1.00, 0.18, 0.10], dtype=np.float32)
PLASTER     = np.array([0.88, 0.86, 0.84], dtype=np.float32)


# =========================================================================== #
#  Icosphere — subdivision 2 for smooth, round point spheres
# =========================================================================== #
def _make_ico(sub=2):
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([(-1,phi,0),(1,phi,0),(-1,-phi,0),(1,-phi,0),
                  (0,-1,phi),(0,1,phi),(0,-1,-phi),(0,1,-phi),
                  (phi,0,-1),(phi,0,1),(-phi,0,-1),(-phi,0,1)], dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
                  (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
                  (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
                  (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)], dtype=np.int32)
    for _ in range(sub):
        em, nf, vl = {}, [], list(v)
        def _m(a, b):
            k = (min(a,b), max(a,b))
            if k in em: return em[k]
            p = (vl[a]+vl[b])/2; p /= np.linalg.norm(p)
            i = len(vl); vl.append(p); em[k] = i; return i
        for t in f:
            a,b,c = int(t[0]),int(t[1]),int(t[2])
            ab,bc,ca = _m(a,b),_m(b,c),_m(c,a)
            nf.extend([(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)])
        v, f = np.array(vl, dtype=np.float64), np.array(nf, dtype=np.int32)
    return v.astype(np.float32), f

_IV, _IF = _make_ico(2)


# =========================================================================== #
#  PLY builder
# =========================================================================== #
def _build_ply(pts, colors, radius):
    tv, tf = _IV, _IF
    ntv, ntf, n = len(tv), len(tf), len(pts)
    all_v = (np.tile(tv, (n, 1)) * radius
             + np.repeat(pts, ntv, 0)).astype(np.float32)
    all_c = np.repeat(colors, ntv, 0).astype(np.float32)
    offs  = np.arange(n, dtype=np.int32)[:, None] * ntv
    all_f = (np.tile(tf, (n, 1))
             + np.repeat(offs, ntf, 0)).astype(np.int32)
    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tot_v, tot_f = n * ntv, n * ntf
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {tot_v}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property float red\nproperty float green\nproperty float blue\n"
           f"element face {tot_f}\n"
           "property list uchar int vertex_indices\nend_header\n")
    with open(tmp.name, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(np.hstack([all_v, all_c]).astype(np.float32).tobytes())
        buf = bytearray(tot_f * 13)
        o = 0
        for i in range(tot_f):
            buf[o] = 3; o += 1
            struct.pack_into("<iii", buf, o,
                             int(all_f[i, 0]), int(all_f[i, 1]), int(all_f[i, 2]))
            o += 12
        f.write(bytes(buf))
    return tmp.name


# =========================================================================== #
#  Mitsuba scene — hide_emitters, no ground, ceramic BSDF
# =========================================================================== #
def _scene(ply, res=(800, 800), spp=256, cam=(2, -2, 1), fov=38):
    return {
        "type": "scene",
        # hide_emitters: area lights contribute illumination but are
        # invisible to the camera → no gray rectangles in the image
        "integrator": {"type": "path", "max_depth": 6,
                       "hide_emitters": True},
        "sensor": {
            "type": "perspective", "fov": fov,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=cam, target=(0, 0, 0), up=(0, 0, 1)),
            "film": {
                "type": "hdrfilm",
                "width": res[0], "height": res[1],
                "pixel_format": "rgba",
                "rfilter": {"type": "gaussian"},
            },
            "sampler": {"type": "independent", "sample_count": spp},
        },
        # Key light — warm, upper right
        "key": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(3, -1, 5), target=(0, 0, 0), up=(0, 0, 1))
                @ mi.ScalarTransform4f.scale((1.0, 1.0, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [9.0, 8.5, 8.0]}},
        },
        # Fill light — cool, left
        "fill": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(-4, 1, 3), target=(0, 0, 0), up=(0, 0, 1))
                @ mi.ScalarTransform4f.scale((2.5, 2.5, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [1.8, 2.0, 2.5]}},
        },
        # Rim light — behind
        "rim": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(-1, 3, 2), target=(0, 0, 0), up=(0, 0, 1))
                @ mi.ScalarTransform4f.scale((1.5, 1.5, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [3.5, 3.5, 3.8]}},
        },
        # AO dome — large soft overhead
        "dome": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 7), target=(0, 0, 0), up=(0, 1, 0))
                @ mi.ScalarTransform4f.scale((6, 6, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [0.5, 0.5, 0.55]}},
        },
        # Bottom bounce
        "bounce": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, -6), target=(0, 0, 0), up=(0, 1, 0))
                @ mi.ScalarTransform4f.scale((4, 4, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [0.25, 0.25, 0.25]}},
        },
        # Point cloud mesh
        "pc": {
            "type": "ply", "filename": ply,
            "bsdf": {
                "type": "twosided",
                "bsdf": {
                    "type": "principled",
                    "base_color": {
                        "type": "mesh_attribute",
                        "name": "vertex_color",
                    },
                    "roughness": 0.40,
                    "specular":  0.35,
                    "metallic":  0.0,
                },
            },
        },
    }


def render(pts, colors, res=(800, 800), spp=256, cam=(2, -2, 1)):
    """Render (N,3) points → RGBA uint8 image."""
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts - pts.mean(0)
    mx = np.abs(pts).max()
    if mx > 1e-6:
        pts = pts * (0.8 / mx)

    # Adaptive sphere radius — slightly larger for solid appearance
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=7)
    radius = float(0.55 * np.median(d[:, 1:]))
    radius = np.clip(radius, 0.005, 0.028)

    colors = np.asarray(colors, dtype=np.float32)
    ply = _build_ply(pts, colors, radius)
    try:
        scene = mi.load_dict(_scene(ply, res, spp, cam))
        raw = np.array(mi.render(scene))
    finally:
        os.unlink(ply)

    if raw.ndim == 3 and raw.shape[2] >= 4:
        rgb   = np.clip(raw[:, :, :3], 0, None)
        alpha = np.clip(raw[:, :, 3:4], 0, 1)
    else:
        rgb = np.clip(raw[:, :, :3], 0, None)
        lum = rgb.max(axis=2, keepdims=True)
        alpha = np.clip(lum / max(float(np.percentile(lum[lum > 0], 1)), 1e-6)
                        * 2, 0, 1) if np.any(lum > 0) else np.zeros_like(lum)

    rgb = rgb / (1.0 + rgb)                           # Reinhard
    rgb = np.power(np.clip(rgb, 0, 1), 1.0 / 2.2)    # sRGB gamma

    rgba = np.concatenate([rgb, alpha], axis=2)
    return (rgba * 255).clip(0, 255).astype(np.uint8)


# =========================================================================== #
#  Coloring helpers
# =========================================================================== #

def color_green_red(pred, gt, threshold_mult=1.8):
    """Green for on-surface, red for noise.  Steep sigmoid transition."""
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)
    dists = dists.astype(np.float32)
    threshold = threshold_mult * float(np.median(dists))
    t = 1.0 / (1.0 + np.exp(-12.0 * (dists / max(threshold, 1e-8) - 1.0)))
    t = t[:, None]
    return VIVID_GREEN * (1.0 - t) + VIVID_RED * t


def color_uniform(n, rgb):
    return np.tile(np.asarray(rgb, dtype=np.float32), (n, 1))


def color_plaster(n):
    return color_uniform(n, PLASTER)


# =========================================================================== #
#  Data & denoising
# =========================================================================== #

def load_shape(name, axis_swap, npoints=0):
    pcl_dir = os.path.join(SHAPENET_DATA, "PUNet",
                           "pointclouds", "test", "10000_poisson")
    pc = np.loadtxt(os.path.join(pcl_dir, f"{name}.xyz"), dtype=np.float32)
    rng = np.random.RandomState(0)
    if npoints > 0 and pc.shape[0] > npoints:
        pc = pc[rng.choice(pc.shape[0], npoints, replace=False)]
    if axis_swap is not None:
        pc = pc[:, axis_swap]
    pc -= pc.mean(0)
    pc /= np.sqrt((pc ** 2).sum(1).max())
    return pc


def bilateral_denoise(pts, k=16, sigma_n=0.5, iters=2):
    pts = pts.copy()
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    sigma_d = float(np.median(d[:, 1:])) * 2.0
    for _ in range(iters):
        tree = cKDTree(pts)
        d, idx = tree.query(pts, k=k + 1)
        nb = pts[idx[:, 1:]]
        diff = nb - pts[:, None, :]
        w = np.exp(-d[:, 1:] ** 2 / (2 * sigma_d ** 2))
        w /= w.sum(1, keepdims=True)
        pts = pts + sigma_n * (diff * w[:, :, None]).sum(1)
    return pts


def sor_denoise(pts, k=20, std_ratio=1.0):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    md = d[:, 1:].mean(1)
    return pts[md < md.mean() + std_ratio * md.std()]


def run_denoising(trainer, clean_dict, noise_std, device):
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss)
    gl = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    names = list(clean_dict.keys())
    batch = torch.stack(
        [torch.from_numpy(clean_dict[n].T).float() for n in names]).to(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    noisy = batch + noise_std * torch.randn(
        batch.shape, device=device, generator=rng)

    noisy_dict = {n: noisy[i].T.cpu().numpy() for i, n in enumerate(names)}
    results = {}

    for label, mtype, lam, ann in METHODS:
        print(f"  [{label}]...", end="", flush=True)
        if mtype == "baseline":
            with torch.no_grad():
                r = ddpm_denoise(trainer.model, trainer.sb_schedule, noisy,
                                 sampling_steps=10, verbose=False)
        elif mtype == "std":
            r = ddpm_denoise_guided(
                trainer.model, trainer.sb_schedule, noisy,
                sampling_steps=10, guidance_scale=lam, annealing=ann,
                geom_loss=gl, grad_clip=2.0, verbose=False)
        else:
            r = ddpm_denoise_ortho_guided(
                trainer.model, trainer.sb_schedule, noisy,
                sampling_steps=10, guidance_scale=lam, annealing=ann,
                geom_loss=gl, grad_clip=2.0, verbose=False)
        results[label] = {
            n: r["x_pred"][i].T.cpu().numpy() for i, n in enumerate(names)}
        print(" done")

    print("  [Bilateral]...", end="", flush=True)
    results["Bilateral"] = {n: bilateral_denoise(noisy_dict[n]) for n in names}
    print(" done")
    print("  [SOR]...", end="", flush=True)
    results["SOR"] = {n: sor_denoise(noisy_dict[n]) for n in names}
    print(" done")

    return noisy_dict, results


# =========================================================================== #
#  Figure 3 — 3×5 subset for main paper body
# =========================================================================== #
FIG3_SHAPES = [
    ("camel",  None,      (1.6, -2.2, 0.6)),   # smooth organic surface
    ("chair",  [0, 2, 1], (1.8, -2.0, 0.8)),   # sharp edges & thin bars
    ("kitten", [0, 2, 1], (1.8, -2.0, 0.5)),   # mixed curvature
]
FIG3_COLS = ["GT", "Noisy", "P2P-Bridge", "Std-0.2", "Orth-0.2"]


def compose_figure3(clean_dict, noisy_dict, results,
                    res=(1200, 1200), spp=256, save_dir=None):
    """3×5 ultra-tight grid for paper Figure 3. No text, gap=1."""
    shape_info = FIG3_SHAPES
    col_labels = FIG3_COLS
    names = [s[0] for s in shape_info]
    cams  = [s[2] for s in shape_info]
    n_rows, n_cols = len(names), len(col_labels)

    print(f"\n[Figure 3] Rendering {n_rows} x {n_cols} = "
          f"{n_rows * n_cols} panels @ {res[0]}px ...")
    panels = [[None] * n_cols for _ in range(n_rows)]

    for ri, name in enumerate(names):
        gt  = clean_dict[name]
        cam = cams[ri]
        print(f"  Row {ri} ({name}): ", end="", flush=True)

        for ci, label in enumerate(col_labels):
            print(f"{label}", end="", flush=True)
            if label == "GT":
                panels[ri][ci] = render(gt, color_green_red(gt, gt),
                                        res, spp, cam)
            elif label == "Noisy":
                noisy_np = noisy_dict[name]
                panels[ri][ci] = render(noisy_np,
                                        color_green_red(noisy_np, gt),
                                        res, spp, cam)
            else:
                pred = results[label][name]
                panels[ri][ci] = render(pred, color_green_red(pred, gt),
                                        res, spp, cam)
            if ci < n_cols - 1:
                print(", ", end="", flush=True)
        print()

    # ── Save individual panels (tight-cropped RGBA) ──
    fig3_dir = os.path.join(save_dir, "figure3") if save_dir else None
    if fig3_dir:
        os.makedirs(fig3_dir, exist_ok=True)
        for ri, name in enumerate(names):
            bbox = _row_bbox(panels[ri], padding=3)
            rmin, rmax, cmin, cmax = bbox
            for ci, label in enumerate(col_labels):
                crop = panels[ri][ci][rmin:rmax+1, cmin:cmax+1]
                safe = label.replace("-", "_").replace(".", "")
                Image.fromarray(crop, "RGBA").save(
                    os.path.join(fig3_dir, f"{name}_{safe}.png"))

    # ── Compose ultra-tight grid: gap=1, white BG ──
    gap = 1
    cropped_rows = []
    for ri in range(n_rows):
        rmin, rmax, cmin, cmax = _row_bbox(panels[ri], padding=3)
        row = [panels[ri][ci][rmin:rmax+1, cmin:cmax+1] for ci in range(n_cols)]
        cropped_rows.append(row)

    max_w = max(cropped_rows[ri][0].shape[1] for ri in range(n_rows))
    row_heights = [cropped_rows[ri][0].shape[0] for ri in range(n_rows)]

    total_w = n_cols * max_w + (n_cols - 1) * gap
    total_h = sum(row_heights) + (n_rows - 1) * gap

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    yo = 0
    for ri in range(n_rows):
        rh = row_heights[ri]
        for ci in range(n_cols):
            panel = cropped_rows[ri][ci]
            ph, pw = panel.shape[:2]
            xo = ci * (max_w + gap)
            x_off = (max_w - pw) // 2
            rgb = _composite_on_white(panel)
            canvas[yo:yo+ph, xo+x_off:xo+x_off+pw] = rgb
        yo += rh + gap

    if save_dir:
        pil = Image.fromarray(canvas)
        png_path = os.path.join(save_dir, "figure3.png")
        pdf_path = os.path.join(save_dir, "figure3.pdf")
        pil.save(png_path, quality=95)
        pil.save(pdf_path, resolution=300)
        print(f"\n[Figure 3] PNG: {png_path}")
        print(f"[Figure 3] PDF: {pdf_path}")
        print(f"[Figure 3] Individual: {fig3_dir}/")

    return canvas


# =========================================================================== #
#  Full 5×7 grid composition — NO TEXT, pure images only
# =========================================================================== #

def _composite_on_white(rgba):
    """RGBA uint8 → RGB uint8 on white."""
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    return (rgb * a + 255.0 * (1.0 - a)).clip(0, 255).astype(np.uint8)


def _row_bbox(panels_row, padding=4):
    """Union bounding box across all RGBA panels in one row."""
    rmin, cmin = 999999, 999999
    rmax, cmax = 0, 0
    for rgba in panels_row:
        alpha = rgba[:, :, 3]
        rows = np.where(np.any(alpha > 10, axis=1))[0]
        cols = np.where(np.any(alpha > 10, axis=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            rmin = min(rmin, rows[0])
            rmax = max(rmax, rows[-1])
            cmin = min(cmin, cols[0])
            cmax = max(cmax, cols[-1])
    h, w = panels_row[0].shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)
    return rmin, rmax, cmin, cmax


def compose(clean_dict, noisy_dict, results, shape_info,
            res=(800, 800), spp=256, save_dir=None, showcase=False):
    """Render all panels → tight-cropped grid, NO text labels."""
    names = [s[0] for s in shape_info]
    cams  = [s[2] for s in shape_info]
    n_rows, n_cols = len(names), len(COL_LABELS)

    indiv_dir = os.path.join(save_dir, "individual") if save_dir else None
    showcase_dir = os.path.join(save_dir, "showcase") if (save_dir and showcase) else None
    if indiv_dir:
        os.makedirs(indiv_dir, exist_ok=True)
    if showcase_dir:
        os.makedirs(showcase_dir, exist_ok=True)

    print(f"\nRendering {n_rows} x {n_cols} = {n_rows * n_cols} panels...")
    panels = [[None] * n_cols for _ in range(n_rows)]

    for ri, name in enumerate(names):
        gt  = clean_dict[name]
        cam = cams[ri]
        print(f"  Row {ri} ({name}): ", end="", flush=True)

        # GT
        print("GT", end="", flush=True)
        panels[ri][0] = render(gt, color_green_red(gt, gt), res, spp, cam)

        # Noisy
        print(", Noisy", end="", flush=True)
        noisy_np = noisy_dict[name]
        panels[ri][1] = render(noisy_np, color_green_red(noisy_np, gt),
                               res, spp, cam)

        # Methods
        for ci, label in enumerate(COL_LABELS[2:], start=2):
            print(f", {label}", end="", flush=True)
            pred = results[label][name]
            panels[ri][ci] = render(pred, color_green_red(pred, gt),
                                    res, spp, cam)
            if showcase_dir:
                rgba_sc = render(pred, color_plaster(len(pred)), res, spp, cam)
                safe = label.replace("-", "_").replace(".", "")
                Image.fromarray(rgba_sc, "RGBA").save(
                    os.path.join(showcase_dir, f"{name}_{safe}_plaster.png"))
        print()

    # ── Save individual tight-cropped PNGs ──
    if indiv_dir:
        for ri, name in enumerate(names):
            bbox = _row_bbox(panels[ri])
            rmin, rmax, cmin, cmax = bbox
            for ci, label in enumerate(COL_LABELS):
                crop = panels[ri][ci][rmin:rmax+1, cmin:cmax+1]
                safe = label.replace("-", "_").replace(".", "")
                Image.fromarray(crop, "RGBA").save(
                    os.path.join(indiv_dir, f"{name}_{safe}.png"))

    # ── Compose tight grid: no text, minimal gap, white BG ──
    gap = 2
    # Per-row tight crop
    cropped_rows = []
    for ri in range(n_rows):
        rmin, rmax, cmin, cmax = _row_bbox(panels[ri])
        row = []
        for ci in range(n_cols):
            row.append(panels[ri][ci][rmin:rmax+1, cmin:cmax+1])
        cropped_rows.append(row)

    # Find max cell width across all rows (for column alignment)
    max_w = max(cropped_rows[ri][0].shape[1] for ri in range(n_rows))
    row_heights = [cropped_rows[ri][0].shape[0] for ri in range(n_rows)]

    total_w = n_cols * max_w + (n_cols - 1) * gap
    total_h = sum(row_heights) + (n_rows - 1) * gap

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    yo = 0
    for ri in range(n_rows):
        rh = row_heights[ri]
        for ci in range(n_cols):
            panel = cropped_rows[ri][ci]
            ph, pw = panel.shape[:2]
            xo = ci * (max_w + gap)
            # Center horizontally if narrower than max_w
            x_off = (max_w - pw) // 2
            rgb = _composite_on_white(panel)
            canvas[yo:yo+ph, xo+x_off:xo+x_off+pw] = rgb
        yo += rh + gap

    if save_dir:
        grid_png = os.path.join(save_dir, "grid_overview.png")
        grid_pdf = os.path.join(save_dir, "grid_overview.pdf")
        pil = Image.fromarray(canvas)
        pil.save(grid_png, quality=95)
        pil.save(grid_pdf, resolution=300)
        print(f"\nGrid PNG: {grid_png}")
        print(f"Grid PDF: {grid_pdf}")
        print(f"Individual: {indiv_dir}/")

    return canvas


# =========================================================================== #
#  Main
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--res", type=int, default=800)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--figure3", action="store_true",
                        help="3x5 main-paper figure (camel/chair/kitten)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Pick which shapes to load
    if args.figure3:
        shapes_to_use = FIG3_SHAPES
    else:
        shapes_to_use = SHAPES

    print("=" * 60)
    mode = "Figure 3 (3x5)" if args.figure3 else "Full grid (5x7)"
    print(f"  Premium Rendering — {mode}")
    print("=" * 60)
    print(f"  sigma={args.noise}  spp={args.spp}  res={args.res}\n")

    print("Loading shapes...")
    clean_dict = {}
    for name, swap, cam in shapes_to_use:
        pc = load_shape(name, swap)
        clean_dict[name] = pc
        print(f"  {name}: {pc.shape}")

    print("\nLoading model...")
    from omegaconf import OmegaConf
    from sb_cover.training.trainer_igv import TrainerIGV
    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    print(f"\nDenoising (sigma={args.noise})...")
    noisy_dict, results = run_denoising(
        trainer, clean_dict, args.noise, device)

    if args.figure3:
        compose_figure3(
            clean_dict, noisy_dict, results,
            res=(args.res, args.res), spp=args.spp,
            save_dir=args.output_dir)
    else:
        compose(
            clean_dict, noisy_dict, results, SHAPES,
            res=(args.res, args.res), spp=args.spp,
            save_dir=args.output_dir, showcase=args.showcase)

    print("Done.")


if __name__ == "__main__":
    main()
