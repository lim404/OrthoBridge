"""Mitsuba 3 visualization of ShapeNet denoising: SB-IGV vs Orthogonal Guidance.

Renders a 5-row x 5-column grid figure:
    Columns: GT | Noisy | SB-IGV | Orth-0.1 | Orth-0.3
    Rows:    5 diverse ShapeNet shapes (Airplane, Airplane, Car, Chair, Chair)

Each panel is colored by per-point NN distance to GT (green=low error, red=high),
except GT (uniform gray) and Noisy (uniform light blue). Error color scale is
shared per row for fair comparison.

Usage:
    conda run -n p2pb python experiments/render_shapenet_comparison.py
    conda run -n p2pb python experiments/render_shapenet_comparison.py --noise 0.03 --spp 128
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
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import mitsuba as mi
mi.set_variant("scalar_rgb")


# =========================================================================== #
#  Constants
# =========================================================================== #

SHAPENET_ROOT = "/mnt/a/Users/Administrator/PycharmProjects/ECCV/data/ShapeNet"

# 5 shapes: pick specific indices for visual diversity
# (synset_id, shape_index, category_name, camera_origin)
# Camera positions tuned for Z-up after Y<->Z swap
SHAPE_SELECTIONS = [
    ("02691156", 2000, "Airplane", (1.8, 2.0, 1.0)),   # wide wingspan
    ("02691156",  200, "Airplane", (1.8, 2.0, 1.0)),   # different airplane
    ("02958343",  200, "Car",      (2.0, 1.8, 1.0)),   # good proportions
    ("03001627",   50, "Chair",    (2.0, 1.5, 1.2)),   # tall chair
    ("03001627", 2000, "Chair",    (2.0, 1.5, 1.2)),   # different chair
]

METHOD_CONFIGS = [
    ("SB-IGV",   "baseline", 0.0, None),
    ("Orth-0.1", "orth",     0.1, "linear_decay"),
    ("Orth-0.3", "orth",     0.3, "linear_decay"),
]

OUTPUT_DIR = "experiments/figures/shapenet_comparison"


# =========================================================================== #
#  Icosphere mesh (for rendering points as small spheres)
# =========================================================================== #

def _make_icosphere(subdivisions=1):
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    verts = np.array(verts, dtype=np.float64)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    faces = np.array(faces, dtype=np.int32)

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        verts_list = list(verts)

        def _get_midpoint(v1, v2):
            key = (min(v1, v2), max(v1, v2))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = (verts_list[v1] + verts_list[v2]) / 2.0
            mid = mid / np.linalg.norm(mid)
            idx = len(verts_list)
            verts_list.append(mid)
            edge_midpoints[key] = idx
            return idx

        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab = _get_midpoint(a, b)
            bc = _get_midpoint(b, c)
            ca = _get_midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])

        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int32)

    return verts.astype(np.float32), faces


_ICO_VERTS, _ICO_FACES = _make_icosphere(subdivisions=1)


# =========================================================================== #
#  PLY generation
# =========================================================================== #

def _build_merged_ply(points, colors, radius):
    """Build a single PLY mesh: all points as colored icospheres."""
    template_v = _ICO_VERTS
    template_f = _ICO_FACES
    n_tv = len(template_v)
    n_tf = len(template_f)
    n_pts = len(points)

    total_verts = n_pts * n_tv
    total_faces = n_pts * n_tf

    tiled_tv = np.tile(template_v, (n_pts, 1))
    centers_rep = np.repeat(points, n_tv, axis=0)
    colors_rep = np.repeat(colors, n_tv, axis=0)

    all_verts = (tiled_tv * radius + centers_rep).astype(np.float32)
    all_colors = colors_rep.astype(np.float32)

    offsets = np.arange(n_pts, dtype=np.int32)[:, None] * n_tv
    tiled_tf = np.tile(template_f, (n_pts, 1))
    face_offsets = np.repeat(offsets, n_tf, axis=0)
    all_faces = (tiled_tf + face_offsets).astype(np.int32)

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {total_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float red\n"
        "property float green\n"
        "property float blue\n"
        f"element face {total_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(tmp.name, "wb") as f:
        f.write(header.encode("ascii"))
        vert_data = np.hstack([all_verts, all_colors]).astype(np.float32)
        f.write(vert_data.tobytes())
        face_buf = bytearray(total_faces * 13)
        offset = 0
        for fi in range(total_faces):
            face_buf[offset] = 3
            offset += 1
            struct.pack_into("<iii", face_buf, offset,
                             int(all_faces[fi, 0]),
                             int(all_faces[fi, 1]),
                             int(all_faces[fi, 2]))
            offset += 12
        f.write(bytes(face_buf))

    return tmp.name


# =========================================================================== #
#  Rendering helpers
# =========================================================================== #

def _adaptive_radius(points, k=6, scale=0.5):
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    median_knn = np.median(dists[:, 1:])
    return float(scale * median_knn)


def _nn_distances(pred, gt):
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)
    return dists.astype(np.float32)


def _error_to_rgb(errors, vmin=0.0, vmax=None, percentile_cap=95.0):
    """Map error to green -> yellow -> red."""
    if vmax is None:
        vmax = float(np.percentile(errors, percentile_cap))
    vmax = max(vmax, vmin + 1e-8)
    t = np.clip((errors - vmin) / (vmax - vmin), 0.0, 1.0)

    c_green = np.array([0.10, 0.78, 0.20])
    c_yellow = np.array([0.95, 0.90, 0.12])
    c_red = np.array([0.85, 0.10, 0.10])

    colors = np.zeros((len(t), 3), dtype=np.float32)
    mask_low = t < 0.5
    t_low = (t[mask_low] * 2.0)[:, None]
    colors[mask_low] = c_green * (1.0 - t_low) + c_yellow * t_low
    mask_high = ~mask_low
    t_high = ((t[mask_high] - 0.5) * 2.0)[:, None]
    colors[mask_high] = c_yellow * (1.0 - t_high) + c_red * t_high
    return colors


def _uniform_color(n, rgb=(0.72, 0.72, 0.72)):
    return np.broadcast_to(np.array(rgb, dtype=np.float32), (n, 3)).copy()


def _build_scene_dict(ply_path, resolution=(600, 600), sample_count=256,
                      camera_origin=None, camera_target=(0, 0, 0), fov=40.0):
    if camera_origin is None:
        camera_origin = (2.0, 1.0, 1.0)

    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective",
            "fov": fov,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=camera_origin, target=camera_target, up=(0, 0, 1)),
            "film": {
                "type": "hdrfilm",
                "width": resolution[0], "height": resolution[1],
                "pixel_format": "rgb",
                "rfilter": {"type": "gaussian"},
            },
            "sampler": {"type": "independent", "sample_count": sample_count},
        },
        "key_light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(3.0, 2.0, 4.0), target=(0, 0, 0), up=(0, 0, 1))
                @ mi.ScalarTransform4f.scale((1.5, 1.5, 1.0)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [6.0, 5.5, 5.0]}},
        },
        "fill_light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(-3.5, -1.0, 2.5), target=(0, 0, 0), up=(0, 0, 1))
                @ mi.ScalarTransform4f.scale((2.0, 2.0, 1.0)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [2.5, 2.8, 3.2]}},
        },
        "top_light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0.0, 0.0, 5.0), target=(0, 0, 0), up=(0, 1, 0))
                @ mi.ScalarTransform4f.scale((2.5, 2.5, 1.0)),
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
            "type": "ply",
            "filename": ply_path,
            "bsdf": {
                "type": "twosided",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "mesh_attribute",
                                    "name": "vertex_color"},
                },
            },
        },
    }


def render_pointcloud(points, colors=None, gt=None, radius=None,
                      resolution=(600, 600), sample_count=256,
                      camera_origin=None, fov=40.0, error_vmax=None):
    """Render a (N, 3) point cloud to an RGB uint8 image."""
    points = np.asarray(points, dtype=np.float32)

    centroid = points.mean(axis=0)
    pts = points - centroid
    max_extent = np.abs(pts).max()
    if max_extent > 1e-6:
        scale = 0.8 / max_extent
        pts = pts * scale
    else:
        scale = 1.0

    if radius is None:
        radius = _adaptive_radius(pts)
    radius = np.clip(radius, 0.004, 0.025)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
    elif gt is not None:
        gt_c = (gt - centroid) * scale
        errors = _nn_distances(pts, gt_c)
        colors = _error_to_rgb(errors, vmax=error_vmax)
    else:
        colors = _uniform_color(len(pts))

    ply_path = _build_merged_ply(pts, colors, radius)
    try:
        scene_dict = _build_scene_dict(
            ply_path, resolution=resolution, sample_count=sample_count,
            camera_origin=camera_origin, fov=fov)
        scene = mi.load_dict(scene_dict)
        image = mi.render(scene)
    finally:
        os.unlink(ply_path)

    img_np = np.array(image)
    img_np = np.clip(img_np, 0, None)
    img_np = img_np / (1.0 + img_np)  # Reinhard tonemap
    img_np = np.power(np.clip(img_np, 0, 1), 1.0 / 2.2)  # sRGB gamma
    return (img_np * 255).clip(0, 255).astype(np.uint8)


# =========================================================================== #
#  Data loading
# =========================================================================== #

def load_shape(synset_id, shape_idx, npoints=2048, seed=0):
    """Load a single ShapeNet shape, subsample, and normalize.

    ShapeNet uses Y-up; we swap Y<->Z so Z is up (matching Mitsuba scene).
    """
    cat_dir = os.path.join(SHAPENET_ROOT, synset_id)
    files = sorted([f for f in os.listdir(cat_dir) if f.endswith(".npy")])
    pc = np.load(os.path.join(cat_dir, files[shape_idx]))  # (8192, 3)

    rng = np.random.RandomState(seed + shape_idx)
    choice = rng.choice(pc.shape[0], npoints, replace=False)
    pc = pc[choice]

    # Swap Y <-> Z: ShapeNet Y-up -> Mitsuba Z-up
    pc = pc[:, [0, 2, 1]]

    centroid = pc.mean(axis=0)
    pc = pc - centroid
    max_r = np.sqrt((pc ** 2).sum(axis=1).max())
    if max_r > 1e-8:
        pc = pc / max_r
    return pc  # (npoints, 3) — Z-up convention


# =========================================================================== #
#  Denoising
# =========================================================================== #

def denoise_shapes(trainer, clean_list, noise_std, device):
    """Denoise a list of clean shapes with all methods.

    Args:
        clean_list: list of (npoints, 3) numpy arrays
        noise_std: noise level
        device: torch device

    Returns:
        noisy_list: list of (npoints, 3) numpy arrays
        results: dict {method_name: list of (npoints, 3) numpy arrays}
    """
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_ortho_guided, GeometricQualityLoss,
    )

    gl = GeometricQualityLoss(
        w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    # Stack into (N, 3, pts) tensor
    batch = torch.stack([
        torch.from_numpy(pc.T).float() for pc in clean_list
    ]).to(device)  # (N, 3, npoints)

    # Add noise
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    noisy = batch + noise_std * torch.randn(batch.shape, device=device, generator=rng)

    noisy_list = [noisy[i].T.cpu().numpy() for i in range(len(clean_list))]

    results = {}
    for name, method_type, lam, annealing in METHOD_CONFIGS:
        print(f"    [{name}]...", end="", flush=True)
        if method_type == "baseline":
            with torch.no_grad():
                r = ddpm_denoise(
                    trainer.model, trainer.sb_schedule,
                    noisy, sampling_steps=10, verbose=False)
            pred = r["x_pred"]
        else:
            r = ddpm_denoise_ortho_guided(
                trainer.model, trainer.sb_schedule,
                noisy, sampling_steps=10,
                guidance_scale=lam, annealing=annealing,
                geom_loss=gl, grad_clip=2.0, verbose=False)
            pred = r["x_pred"]

        results[name] = [pred[i].T.cpu().numpy() for i in range(len(clean_list))]
        print(" done")

    return noisy_list, results


# =========================================================================== #
#  Figure composition
# =========================================================================== #

def _try_load_font(size=28):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _density_to_rgb(points, k=8):
    """Color by local density uniformity.

    Green = uniform local density (good), red = too dense (clustering).
    Uses coefficient of variation of k-NN distances.
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    knn_dists = dists[:, 1:]  # exclude self
    mean_d = knn_dists.mean(axis=1)
    global_median = np.median(mean_d)

    # Density ratio: >1 means too clustered, <1 means too sparse
    ratio = global_median / np.clip(mean_d, 1e-8, None)
    # Map: ratio~1 = green, ratio>>1 = red (dense), ratio<<1 = blue (sparse)
    deviation = np.abs(np.log(np.clip(ratio, 0.1, 10.0)))
    # Normalize
    vmax = float(np.percentile(deviation, 95))
    t = np.clip(deviation / max(vmax, 1e-8), 0.0, 1.0)

    c_green = np.array([0.10, 0.78, 0.20])
    c_yellow = np.array([0.95, 0.85, 0.12])
    c_red = np.array([0.85, 0.10, 0.10])

    colors = np.zeros((len(t), 3), dtype=np.float32)
    mask_low = t < 0.5
    t_low = (t[mask_low] * 2.0)[:, None]
    colors[mask_low] = c_green * (1.0 - t_low) + c_yellow * t_low
    mask_high = ~mask_low
    t_high = ((t[mask_high] - 0.5) * 2.0)[:, None]
    colors[mask_high] = c_yellow * (1.0 - t_high) + c_red * t_high
    return colors


def compose_grid(
    clean_list, noisy_list, method_results, shape_names, camera_origins,
    resolution=(600, 600), sample_count=256, save_path=None,
    color_mode="error",
):
    """Compose a grid figure: rows=shapes, cols=GT|Noisy|methods.

    Args:
        color_mode: "error" for NN-distance error heatmap,
                    "density" for local density uniformity,
                    "uniform" for same color on all.
    """
    method_names = [mc[0] for mc in METHOD_CONFIGS]
    col_labels = ["GT", "Noisy"] + method_names
    n_rows = len(clean_list)
    n_cols = len(col_labels)

    # Compute metrics for annotations
    metrics = {}
    for row_i in range(n_rows):
        gt_pts = clean_list[row_i]
        gt_t = torch.from_numpy(gt_pts).float()
        for mname in method_names:
            pred_pts = method_results[mname][row_i]
            d1 = _nn_distances(pred_pts, gt_pts).mean()
            d2 = _nn_distances(gt_pts, pred_pts).mean()
            cd = (d1 + d2) * 1000
            # VD: local density deviation
            from metrics.geometric_metrics import compute_vd
            pred_t = torch.from_numpy(pred_pts).float()
            vd = compute_vd(pred_t, gt_t)
            metrics[(row_i, mname)] = {"CD": cd, "VD": vd}

    # Render all panels
    print(f"\nRendering {n_rows} x {n_cols} = {n_rows * n_cols} panels "
          f"(color_mode={color_mode})...")
    panels = [[None] * n_cols for _ in range(n_rows)]

    for row_i in range(n_rows):
        gt_pts = clean_list[row_i]
        noisy_pts = noisy_list[row_i]
        cam = camera_origins[row_i]

        # Shared error vmax for error mode
        if color_mode == "error":
            all_errors = []
            for mname in method_names:
                pred_pts = method_results[mname][row_i]
                errs = _nn_distances(pred_pts, gt_pts)
                all_errors.append(errs)
            shared_vmax = float(np.percentile(np.concatenate(all_errors), 95.0))

        # GT panel — always uniform light gray
        print(f"  Row {row_i} ({shape_names[row_i]}): GT", end="", flush=True)
        panels[row_i][0] = render_pointcloud(
            gt_pts, colors=_uniform_color(len(gt_pts), rgb=(0.72, 0.72, 0.72)),
            resolution=resolution, sample_count=sample_count,
            camera_origin=cam)

        # Noisy panel — uniform blue tint
        print(", Noisy", end="", flush=True)
        panels[row_i][1] = render_pointcloud(
            noisy_pts, colors=_uniform_color(len(noisy_pts), rgb=(0.55, 0.65, 0.80)),
            resolution=resolution, sample_count=sample_count,
            camera_origin=cam)

        # Method panels
        for col_j, mname in enumerate(method_names):
            print(f", {mname}", end="", flush=True)
            pred_pts = method_results[mname][row_i]

            if color_mode == "error":
                panels[row_i][2 + col_j] = render_pointcloud(
                    pred_pts, gt=gt_pts,
                    resolution=resolution, sample_count=sample_count,
                    camera_origin=cam, error_vmax=shared_vmax)
            elif color_mode == "density":
                colors = _density_to_rgb(pred_pts)
                panels[row_i][2 + col_j] = render_pointcloud(
                    pred_pts, colors=colors,
                    resolution=resolution, sample_count=sample_count,
                    camera_origin=cam)
            else:  # uniform
                panels[row_i][2 + col_j] = render_pointcloud(
                    pred_pts,
                    colors=_uniform_color(len(pred_pts), rgb=(0.72, 0.72, 0.72)),
                    resolution=resolution, sample_count=sample_count,
                    camera_origin=cam)

        print()

    # Compose into single image
    panel_h, panel_w = panels[0][0].shape[:2]
    title_h = 60
    row_label_w = 120
    gap = 4
    has_colorbar = color_mode in ("error", "density")
    colorbar_w = 60 if has_colorbar else 10

    total_w = row_label_w + n_cols * panel_w + (n_cols - 1) * gap + colorbar_w
    total_h = title_h + n_rows * (panel_h + 30) + (n_rows - 1) * gap
    # 30 extra per row for metric text below

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    title_font = _try_load_font(26)
    label_font = _try_load_font(20)
    metric_font = _try_load_font(15)

    # Column headers
    for col_j, label in enumerate(col_labels):
        x_center = row_label_w + col_j * (panel_w + gap) + panel_w // 2
        bbox = draw.textbbox((0, 0), label, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text((x_center - tw // 2, 15), label, fill=(30, 30, 30), font=title_font)

    for row_i in range(n_rows):
        gt_pts = clean_list[row_i]
        y_offset = title_h + row_i * (panel_h + 30 + gap)

        # Row label
        rlabel = shape_names[row_i]
        bbox = draw.textbbox((0, 0), rlabel, font=label_font)
        rh = bbox[3] - bbox[1]
        draw.text((10, y_offset + panel_h // 2 - rh // 2), rlabel,
                  fill=(60, 60, 60), font=label_font)

        for col_j in range(n_cols):
            x_offset = row_label_w + col_j * (panel_w + gap)
            panel_img = Image.fromarray(panels[row_i][col_j])
            pil_canvas.paste(panel_img, (x_offset, y_offset))

            # Metric text below panel
            if col_j >= 2:
                mname = method_names[col_j - 2]
                m = metrics[(row_i, mname)]
                metric_text = f"CD={m['CD']:.1f}  VD={m['VD']:.4f}"
                bbox = draw.textbbox((0, 0), metric_text, font=metric_font)
                tw = bbox[2] - bbox[0]
                draw.text(
                    (x_offset + panel_w // 2 - tw // 2, y_offset + panel_h + 4),
                    metric_text, fill=(60, 60, 60), font=metric_font)

    # Colorbar (only for error/density modes)
    if has_colorbar:
        cb_x = row_label_w + n_cols * (panel_w + gap) + 5
        cb_y = title_h + 30
        cb_h = total_h - title_h - 60
        cb_w = 18
        for row in range(cb_h):
            t = row / max(cb_h - 1, 1)
            rgb = _error_to_rgb(np.array([t]), vmin=0.0, vmax=1.0)[0]
            r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            draw.rectangle([cb_x, cb_y + row, cb_x + cb_w, cb_y + row + 1],
                           fill=(r, g, b))
        if color_mode == "error":
            draw.text((cb_x - 5, cb_y - 18), "Low", fill=(60, 60, 60), font=metric_font)
            draw.text((cb_x - 8, cb_y + cb_h + 4), "High", fill=(60, 60, 60), font=metric_font)
        else:
            draw.text((cb_x - 22, cb_y - 18), "Uniform", fill=(60, 60, 60), font=metric_font)
            draw.text((cb_x - 18, cb_y + cb_h + 4), "Clust.", fill=(60, 60, 60), font=metric_font)

    result = np.array(pil_canvas)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pil_canvas.save(save_path, quality=95)
        print(f"\nSaved: {save_path}")

    return result


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Render ShapeNet denoising comparison with Mitsuba")
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Noise std (default: 0.02)")
    parser.add_argument("--spp", type=int, default=256,
                        help="Samples per pixel (default: 256)")
    parser.add_argument("--res", type=int, default=600,
                        help="Per-panel resolution (default: 600)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str,
                        default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--color_mode", type=str, default="all",
                        choices=["error", "density", "uniform", "all"],
                        help="Coloring: error (NN-dist), density (uniformity), "
                             "uniform (gray), or all three (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load shapes
    print("=" * 70)
    print("  Mitsuba ShapeNet Comparison Rendering")
    print("=" * 70)
    print(f"  Noise: {args.noise}, SPP: {args.spp}, Resolution: {args.res}")
    print()

    print("Loading shapes...")
    clean_list = []
    shape_names = []
    camera_origins = []
    for synset_id, shape_idx, cat_name, cam in SHAPE_SELECTIONS:
        pc = load_shape(synset_id, shape_idx)
        clean_list.append(pc)
        shape_names.append(cat_name)
        camera_origins.append(cam)
        print(f"  {cat_name} ({synset_id}[{shape_idx}]): {pc.shape}")

    # Load model and denoise
    print("\nLoading model...")
    from omegaconf import OmegaConf
    from sb_cover.training.trainer_igv import TrainerIGV

    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    print(f"\nDenoising {len(clean_list)} shapes (sigma={args.noise})...")
    noisy_list, method_results = denoise_shapes(
        trainer, clean_list, args.noise, device)

    # Determine which color modes to render
    modes = ["error", "density", "uniform"] if args.color_mode == "all" else [args.color_mode]

    for cmode in modes:
        save_path = os.path.join(
            args.output_dir,
            f"shapenet_{cmode}_sigma{args.noise:.3f}.png")

        compose_grid(
            clean_list=clean_list,
            noisy_list=noisy_list,
            method_results=method_results,
            shape_names=shape_names,
            camera_origins=camera_origins,
            resolution=(args.res, args.res),
            sample_count=args.spp,
            save_path=save_path,
            color_mode=cmode,
        )

    print(f"\nAll renders saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
