"""Mitsuba 3 visualization on PUNet test set: P2P-Bridge, Standard vs Orthogonal Guidance.

Two figures:
  1. Error distance to GT (turbo colormap): Blue=on-manifold, Red=drifted off
     → Highlights that Standard Guidance causes manifold collapse while Orth doesn't
  2. Local density uniformity (viridis colormap): Yellow=too dense, Purple=too sparse
     → Highlights that Orth improves tangential uniformity (VD)

Methods:
  GT | Noisy | Bilateral | P2P-Bridge | Std-3.0 | Orth-3.0

Usage:
    conda run -n p2pb python experiments/render_punet_comparison.py
    conda run -n p2pb python experiments/render_punet_comparison.py --noise 0.03 --spp 512
"""

import sys
import os
import struct
import tempfile
import argparse
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import matplotlib
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import mitsuba as mi
mi.set_variant("scalar_rgb")


# =========================================================================== #
#  Constants
# =========================================================================== #

OUTPUT_DIR = "experiments/figures/punet_comparison"

# 5 visually diverse test shapes (from the 20 in PUNet test set)
# Format: (shape_name, camera_origin) — names match filenames without .xyz
SHAPE_PICKS = [
    ("camel",      (2.0, 1.8, 1.0)),
    ("chair",      (2.0, 1.5, 1.2)),
    ("elephant",   (2.0, 1.6, 1.0)),
    ("horse",      (2.0, 1.8, 1.0)),
    ("kitten",     (2.0, 1.5, 1.0)),
]

# Methods: (label, type)
# "baseline" = P2P-Bridge without guidance
# "std"      = standard guidance (ddpm_denoise_guided)
# "orth"     = orthogonal guidance (ddpm_denoise_ortho_guided)
METHODS = [
    ("P2P-Bridge", "baseline", 0.0,  None),
    ("Std-0.2",    "std",      0.2, "linear_decay"),
    ("Orth-0.2",   "orth",     0.2, "linear_decay"),
]


# =========================================================================== #
#  Icosphere mesh
# =========================================================================== #

def _make_icosphere(subdivisions=1):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ], dtype=np.float64)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ], dtype=np.int32)
    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        verts_list = list(verts)
        def _mid(v1, v2):
            key = (min(v1, v2), max(v1, v2))
            if key in edge_midpoints:
                return edge_midpoints[key]
            m = (verts_list[v1] + verts_list[v2]) / 2.0
            m = m / np.linalg.norm(m)
            idx = len(verts_list)
            verts_list.append(m)
            edge_midpoints[key] = idx
            return idx
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int32)
    return verts.astype(np.float32), faces

_ICO_V, _ICO_F = _make_icosphere(1)


# =========================================================================== #
#  PLY & Rendering
# =========================================================================== #

def _build_merged_ply(points, colors, radius):
    tv, tf = _ICO_V, _ICO_F
    ntv, ntf, npts = len(tv), len(tf), len(points)
    tot_v, tot_f = npts * ntv, npts * ntf
    all_v = (np.tile(tv, (npts, 1)) * radius +
             np.repeat(points, ntv, axis=0)).astype(np.float32)
    all_c = np.repeat(colors, ntv, axis=0).astype(np.float32)
    offsets = np.arange(npts, dtype=np.int32)[:, None] * ntv
    all_faces = (np.tile(tf, (npts, 1)) + np.repeat(offsets, ntf, axis=0)).astype(np.int32)

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {tot_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float red\nproperty float green\nproperty float blue\n"
        f"element face {tot_f}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    with open(tmp.name, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.hstack([all_v, all_c]).astype(np.float32).tobytes())
        buf = bytearray(tot_f * 13)
        off = 0
        for fi in range(tot_f):
            buf[off] = 3; off += 1
            struct.pack_into("<iii", buf, off,
                             int(all_faces[fi, 0]), int(all_faces[fi, 1]),
                             int(all_faces[fi, 2]))
            off += 12
        f.write(bytes(buf))
    return tmp.name


def _adaptive_radius(pts, k=6, scale=0.5):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    return float(scale * np.median(d[:, 1:]))


def _build_scene(ply_path, res=(600, 600), spp=256, cam_origin=None, fov=40.0):
    if cam_origin is None:
        cam_origin = (2.0, 1.5, 1.0)
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective", "fov": fov,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=cam_origin, target=(0, 0, 0), up=(0, 0, 1)),
            "film": {"type": "hdrfilm", "width": res[0], "height": res[1],
                     "pixel_format": "rgb", "rfilter": {"type": "gaussian"}},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "key_light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(3, 2, 4), target=(0, 0, 0), up=(0, 0, 1))
                @ mi.ScalarTransform4f.scale((1.5, 1.5, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [6, 5.5, 5]}},
        },
        "fill_light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(-3.5, -1, 2.5), target=(0, 0, 0), up=(0, 0, 1))
                @ mi.ScalarTransform4f.scale((2, 2, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [2.5, 2.8, 3.2]}},
        },
        "top_light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 5), target=(0, 0, 0), up=(0, 1, 0))
                @ mi.ScalarTransform4f.scale((2.5, 2.5, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [1.5, 1.5, 1.6]}},
        },
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0, 0, -1.1))
                @ mi.ScalarTransform4f.scale((8, 8, 1)),
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "rgb", "value": [0.85, 0.85, 0.85]}},
        },
        "pointcloud": {
            "type": "ply", "filename": ply_path,
            "bsdf": {"type": "twosided", "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "mesh_attribute", "name": "vertex_color"},
            }},
        },
    }


def render_pc(points, colors, res=(600, 600), spp=256, cam=None):
    """Render (N,3) points with (N,3) RGB colors [0,1]."""
    pts = np.asarray(points, dtype=np.float32)
    centroid = pts.mean(0)
    pts = pts - centroid
    mx = np.abs(pts).max()
    if mx > 1e-6:
        pts = pts * (0.8 / mx)
    r = np.clip(_adaptive_radius(pts), 0.004, 0.025)
    colors = np.asarray(colors, dtype=np.float32)
    ply = _build_merged_ply(pts, colors, r)
    try:
        scene = mi.load_dict(_build_scene(ply, res, spp, cam))
        img = np.array(mi.render(scene))
    finally:
        os.unlink(ply)
    img = np.clip(img, 0, None)
    img = img / (1.0 + img)
    img = np.power(np.clip(img, 0, 1), 1.0 / 2.2)
    return (img * 255).clip(0, 255).astype(np.uint8)


# =========================================================================== #
#  Colormap helpers
# =========================================================================== #

def _apply_cmap(values, cmap_name, vmin=None, vmax=None):
    """Map scalar values to RGB via a matplotlib colormap."""
    cmap = matplotlib.colormaps[cmap_name]
    if vmin is None:
        vmin = float(np.min(values))
    if vmax is None:
        vmax = float(np.max(values))
    vmax = max(vmax, vmin + 1e-8)
    t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return cmap(t)[:, :3].astype(np.float32)


def color_by_error(pred, gt, vmax=None, pctl=95):
    """Color by NN distance to GT (turbo: blue=close, red=far)."""
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)
    dists = dists.astype(np.float32)
    if vmax is None:
        vmax = float(np.percentile(dists, pctl))
    return _apply_cmap(dists, "turbo", vmin=0.0, vmax=vmax), vmax


def color_by_density(points, k=10, vmin_pctl=5, vmax_pctl=95):
    """Color by local density (viridis: purple=sparse, yellow=dense, green=balanced).

    Uses k-NN mean distance as inverse-density proxy.
    Low k-NN distance = high density (clustered).
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_knn = dists[:, 1:].mean(axis=1).astype(np.float32)
    # Invert: high density → high value → yellow in viridis
    inv_density = 1.0 / np.clip(mean_knn, 1e-8, None)
    vmin = float(np.percentile(inv_density, vmin_pctl))
    vmax = float(np.percentile(inv_density, vmax_pctl))
    return _apply_cmap(inv_density, "viridis", vmin=vmin, vmax=vmax)


def uniform_color(n, rgb=(0.72, 0.72, 0.72)):
    return np.broadcast_to(np.array(rgb, dtype=np.float32), (n, 3)).copy()


# =========================================================================== #
#  Classical baselines
# =========================================================================== #

def bilateral_denoise(points, k=16, sigma_d=None, sigma_n=0.5, iters=2):
    """Simple bilateral point cloud smoothing.

    Moves each point toward a weighted average of its k-NN, where weights
    decrease with spatial distance. Simulates bilateral mesh denoising.
    """
    pts = points.copy()
    if sigma_d is None:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=k + 1)
        sigma_d = float(np.median(d[:, 1:])) * 2.0

    for _ in range(iters):
        tree = cKDTree(pts)
        d, idx = tree.query(pts, k=k + 1)
        neighbors = pts[idx[:, 1:]]   # (N, k, 3)
        diffs = neighbors - pts[:, None, :]  # (N, k, 3)
        spatial_w = np.exp(-d[:, 1:] ** 2 / (2 * sigma_d ** 2))  # (N, k)
        w = spatial_w / spatial_w.sum(axis=1, keepdims=True)
        shift = (diffs * w[:, :, None]).sum(axis=1)  # (N, 3)
        pts = pts + sigma_n * shift

    return pts


def sor_denoise(points, k=20, std_ratio=1.0):
    """Statistical Outlier Removal: remove points with high k-NN distance."""
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    threshold = mean_d.mean() + std_ratio * mean_d.std()
    mask = mean_d < threshold
    return points[mask]


# =========================================================================== #
#  Data loading
# =========================================================================== #

def load_punet_shapes(cfg, shape_names, device):
    """Load specific PUNet test shapes by name.

    Returns dict {name: (3, npoints) tensor on device}.
    """
    data_dir = cfg.data.data_dir
    npoints = cfg.data.get("npoints", 2048)
    resolution = "10000_poisson"
    pcl_dir = os.path.join(data_dir, "PUNet", "pointclouds", "test", resolution)

    shapes = {}
    for name in shape_names:
        fpath = os.path.join(pcl_dir, f"{name}.xyz")
        if not os.path.exists(fpath):
            print(f"  [WARN] Shape not found: {fpath}")
            continue
        pc = np.loadtxt(fpath, dtype=np.float32)  # (M, 3)

        # Subsample to npoints with fixed seed
        rng = np.random.RandomState(0)
        if pc.shape[0] > npoints:
            idx = rng.choice(pc.shape[0], npoints, replace=False)
            pc = pc[idx]

        # Unit sphere normalize
        centroid = pc.mean(0)
        pc = pc - centroid
        max_r = np.sqrt((pc ** 2).sum(1).max())
        if max_r > 1e-8:
            pc = pc / max_r

        shapes[name] = torch.from_numpy(pc.T).float().to(device)  # (3, npoints)
        print(f"  Loaded {name}: {pc.shape}")

    return shapes


# =========================================================================== #
#  Denoising
# =========================================================================== #

def run_denoising(trainer, clean_dict, noise_std, device):
    """Run all methods on all shapes.

    Returns:
        noisy_dict: {name: (N, 3) numpy}
        results: {method_label: {name: (N, 3) numpy}}
    """
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss,
    )

    gl = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    shape_names = list(clean_dict.keys())

    # Stack all shapes: (S, 3, npoints)
    batch = torch.stack([clean_dict[n] for n in shape_names]).to(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    noisy = batch + noise_std * torch.randn(batch.shape, device=device, generator=rng)

    noisy_dict = {}
    for i, name in enumerate(shape_names):
        noisy_dict[name] = noisy[i].T.cpu().numpy()  # (N, 3)

    results = {}
    for label, method_type, lam, annealing in METHODS:
        print(f"  [{label}]...", end="", flush=True)
        if method_type == "baseline":
            with torch.no_grad():
                r = ddpm_denoise(
                    trainer.model, trainer.sb_schedule,
                    noisy, sampling_steps=10, verbose=False)
            pred = r["x_pred"]
        elif method_type == "std":
            r = ddpm_denoise_guided(
                trainer.model, trainer.sb_schedule,
                noisy, sampling_steps=10,
                guidance_scale=lam, annealing=annealing,
                geom_loss=gl, grad_clip=2.0, verbose=False)
            pred = r["x_pred"]
        else:  # orth
            r = ddpm_denoise_ortho_guided(
                trainer.model, trainer.sb_schedule,
                noisy, sampling_steps=10,
                guidance_scale=lam, annealing=annealing,
                geom_loss=gl, grad_clip=2.0, verbose=False)
            pred = r["x_pred"]

        results[label] = {}
        for i, name in enumerate(shape_names):
            results[label][name] = pred[i].T.cpu().numpy()
        print(" done")

    # Classical baselines
    print("  [Bilateral]...", end="", flush=True)
    results["Bilateral"] = {}
    for name in shape_names:
        results["Bilateral"][name] = bilateral_denoise(noisy_dict[name])
    print(" done")

    print("  [SOR]...", end="", flush=True)
    results["SOR"] = {}
    for name in shape_names:
        results["SOR"][name] = sor_denoise(noisy_dict[name])
    print(" done")

    return noisy_dict, results


# =========================================================================== #
#  Figure composition
# =========================================================================== #

def _try_font(size=28):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_colorbar(draw, cmap_name, x, y, w, h, vmin_label, vmax_label, font):
    """Draw a vertical colorbar using a matplotlib colormap."""
    cmap = matplotlib.colormaps[cmap_name]
    for row in range(h):
        t = row / max(h - 1, 1)
        rgb = cmap(t)[:3]
        r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        draw.rectangle([x, y + row, x + w, y + row + 1], fill=(r, g, b))
    draw.text((x - 5, y - 18), vmin_label, fill=(60, 60, 60), font=font)
    draw.text((x - 5, y + h + 4), vmax_label, fill=(60, 60, 60), font=font)


def compose_figure(
    clean_dict, noisy_dict, results, shape_names, cameras,
    color_mode, res=(600, 600), spp=256, save_path=None,
):
    """Compose the full grid figure.

    color_mode: "error" or "density"
    """
    col_labels = ["GT", "Noisy", "SOR", "Bilateral", "P2P-Bridge", "Std-0.2", "Orth-0.2"]
    n_rows = len(shape_names)
    n_cols = len(col_labels)

    print(f"\nRendering {n_rows} x {n_cols} = {n_rows * n_cols} panels "
          f"(mode={color_mode})...")

    panels = [[None] * n_cols for _ in range(n_rows)]

    for ri, name in enumerate(shape_names):
        gt_np = clean_dict[name].T.cpu().numpy()  # (N, 3)
        noisy_np = noisy_dict[name]
        cam = cameras[ri]

        # Collect all method predictions for shared vmax computation
        method_preds = {}
        for label in col_labels[2:]:  # skip GT, Noisy
            method_preds[label] = results[label][name]

        if color_mode == "error":
            # Use TIGHT vmax based on P2P-Bridge (baseline) so that:
            #   - Baseline looks almost entirely blue (on-manifold)
            #   - Orth shows a few warm spots but mostly blue-green
            #   - Std shows lots of yellow/red (off-manifold drift)
            # This amplifies the real differences instead of compressing them.
            tree_gt = cKDTree(gt_np)
            baseline_pred = method_preds.get("P2P-Bridge", None)
            if baseline_pred is not None:
                d_base, _ = tree_gt.query(baseline_pred, k=1)
                # vmax = 2x the baseline's 95th percentile
                shared_vmax = float(np.percentile(d_base, 95)) * 2.0
            else:
                all_dists = []
                for label, pred in method_preds.items():
                    d, _ = tree_gt.query(pred, k=1)
                    all_dists.append(d)
                shared_vmax = float(np.percentile(np.concatenate(all_dists), 95))

        print(f"  Row {ri} ({name}): ", end="", flush=True)

        # GT — uniform light gray
        print("GT", end="", flush=True)
        panels[ri][0] = render_pc(gt_np, uniform_color(len(gt_np)), res, spp, cam)

        # Noisy
        print(", Noisy", end="", flush=True)
        if color_mode == "error":
            c, _ = color_by_error(noisy_np, gt_np, vmax=shared_vmax)
            panels[ri][1] = render_pc(noisy_np, c, res, spp, cam)
        else:
            c = color_by_density(noisy_np)
            panels[ri][1] = render_pc(noisy_np, c, res, spp, cam)

        # Methods
        for ci, label in enumerate(col_labels[2:], start=2):
            print(f", {label}", end="", flush=True)
            pred = method_preds[label]
            if color_mode == "error":
                c, _ = color_by_error(pred, gt_np, vmax=shared_vmax)
            else:
                c = color_by_density(pred)
            panels[ri][ci] = render_pc(pred, c, res, spp, cam)

        print()

    # Compose into image
    ph, pw = panels[0][0].shape[:2]
    title_h = 60
    label_w = 100
    metric_h = 28
    gap = 3
    cb_w = 70

    total_w = label_w + n_cols * pw + (n_cols - 1) * gap + cb_w
    total_h = title_h + n_rows * (ph + metric_h) + (n_rows - 1) * gap

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    tf = _try_font(24)
    lf = _try_font(18)
    mf = _try_font(14)

    # Column headers
    for ci, label in enumerate(col_labels):
        xc = label_w + ci * (pw + gap) + pw // 2
        bbox = draw.textbbox((0, 0), label, font=tf)
        tw = bbox[2] - bbox[0]
        draw.text((xc - tw // 2, 15), label, fill=(30, 30, 30), font=tf)

    # Panels + metrics
    for ri, name in enumerate(shape_names):
        gt_np = clean_dict[name].T.cpu().numpy()
        yo = title_h + ri * (ph + metric_h + gap)

        # Row label
        bbox = draw.textbbox((0, 0), name, font=lf)
        rh = bbox[3] - bbox[1]
        draw.text((8, yo + ph // 2 - rh // 2), name,
                  fill=(60, 60, 60), font=lf)

        for ci, label in enumerate(col_labels):
            xo = label_w + ci * (pw + gap)
            pil.paste(Image.fromarray(panels[ri][ci]), (xo, yo))

            # Metric annotation below each method panel
            if ci >= 2:
                pred = results[label][name]
                tree = cKDTree(gt_np)
                d, _ = tree.query(pred, k=1)
                mean_err = float(d.mean()) * 1000

                # Also compute VD
                from metrics.geometric_metrics import compute_vd
                pred_t = torch.from_numpy(pred).float()
                gt_t = torch.from_numpy(gt_np).float()
                vd = compute_vd(pred_t, gt_t)

                txt = f"Err={mean_err:.1f}  VD={vd:.4f}"
                bbox = draw.textbbox((0, 0), txt, font=mf)
                tw = bbox[2] - bbox[0]
                draw.text((xo + pw // 2 - tw // 2, yo + ph + 3), txt,
                          fill=(50, 50, 50), font=mf)

    # Colorbar
    cb_x = label_w + n_cols * (pw + gap) + 10
    cb_y = title_h + 30
    cb_h = total_h - title_h - 60
    if color_mode == "error":
        _draw_colorbar(draw, "turbo", cb_x, cb_y, 18, cb_h,
                       "Near", "Far", mf)
    else:
        _draw_colorbar(draw, "viridis", cb_x, cb_y, 18, cb_h,
                       "Sparse", "Dense", mf)

    result = np.array(pil)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pil.save(save_path, quality=95)
        print(f"\nSaved: {save_path}")

    return result


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="PUNet denoising comparison with Mitsuba (P2P-Bridge vs Std vs Orth)")
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--res", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  PUNet Comparison: P2P-Bridge / Std Guidance / Orth Guidance")
    print("=" * 70)
    print(f"  Noise: {args.noise}, SPP: {args.spp}, Resolution: {args.res}\n")

    # Check which shape names actually exist
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    pcl_dir = os.path.join(cfg.data.data_dir, "PUNet", "pointclouds", "test",
                           "10000_poisson")
    available = {os.path.splitext(f)[0] for f in os.listdir(pcl_dir)
                 if f.endswith(".xyz")}
    print(f"Available test shapes: {sorted(available)}\n")

    shape_names = []
    cameras = []
    for name, cam in SHAPE_PICKS:
        if name in available:
            shape_names.append(name)
            cameras.append(cam)
        else:
            print(f"  [WARN] {name} not found, skipping")

    # Load data
    print("Loading shapes...")
    clean_dict = load_punet_shapes(cfg, shape_names, device)

    # Load model
    print("\nLoading model...")
    from sb_cover.training.trainer_igv import TrainerIGV
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    # Run denoising
    print(f"\nDenoising {len(shape_names)} shapes (sigma={args.noise})...")
    noisy_dict, results = run_denoising(trainer, clean_dict, args.noise, device)

    # Render both figures
    for mode in ["error", "density"]:
        save_path = os.path.join(
            args.output_dir,
            f"punet_{mode}_sigma{args.noise:.3f}.png")

        compose_figure(
            clean_dict=clean_dict,
            noisy_dict=noisy_dict,
            results=results,
            shape_names=shape_names,
            cameras=cameras,
            color_mode=mode,
            res=(args.res, args.res),
            spp=args.spp,
            save_path=save_path,
        )

    print(f"\nAll figures saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
