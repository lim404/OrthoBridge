"""High-quality Mitsuba 3 visualization — clay/ceramic material with AO.

Surface points = matte green ceramic.  Outlier/noise points = red.
Roughplastic BSDF with directional shadows and ambient occlusion.

Columns: GT | Noisy | SOR | Bilateral | P2P-Bridge | Std-0.2 | Orth-0.2
Rows:    5 PUNet test shapes

Usage:
    conda run -n p2pb python experiments/render_punet_clean.py
    conda run -n p2pb python experiments/render_punet_clean.py --noise 0.03 --spp 512
"""

import sys, os, struct, tempfile, argparse, time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import mitsuba as mi
mi.set_variant("scalar_rgb")

# =========================================================================== #
SHAPENET_DATA = "/mnt/a/Users/Administrator/PycharmProjects/ECCV/data/objects"
OUTPUT_DIR    = "experiments/figures/punet_clean"

# (name, axis_swap, camera)
#   axis_swap: permutation to bring shape into Z-up, front-facing
#   We checked: camel/horse have tallest=Z already; chair/elephant/kitten tallest=Y
#   For Y-up shapes we swap Y<->Z so tallest axis becomes Z (up).
#   camera: (origin) looking at (0,0,0), up=(0,0,1)
SHAPES = [
    ("camel",    None,       (1.6, -2.2, 0.6)),
    ("horse",    None,       (1.6, -2.2, 0.6)),
    ("chair",    [0, 2, 1], (1.8, -2.0, 0.8)),
    ("elephant", [0, 2, 1], (1.8, -2.0, 0.5)),
    ("kitten",   [0, 2, 1], (1.8, -2.0, 0.5)),
]

METHODS = [
    ("P2P-Bridge", "baseline", 0.0,  None),
    ("Std-0.2",    "std",      0.2,  "linear_decay"),
    ("Orth-0.2",   "orth",     0.2,  "linear_decay"),
]

COL_LABELS = ["GT", "Noisy", "SOR", "Bilateral", "P2P-Bridge", "Std-0.2", "Orth-0.2"]

# Colors
GREEN  = np.array([0.35, 0.72, 0.38], dtype=np.float32)   # jade ceramic
RED    = np.array([0.82, 0.12, 0.12], dtype=np.float32)   # noise
GRAY   = np.array([0.78, 0.76, 0.73], dtype=np.float32)   # GT plaster


# =========================================================================== #
#  Icosphere
# =========================================================================== #
def _make_ico(sub=1):
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

_IV, _IF = _make_ico(1)

# =========================================================================== #
#  PLY with per-vertex color
# =========================================================================== #
def _build_ply(pts, colors, radius):
    tv, tf = _IV, _IF
    ntv, ntf, n = len(tv), len(tf), len(pts)
    all_v = (np.tile(tv,(n,1))*radius + np.repeat(pts,ntv,0)).astype(np.float32)
    all_c = np.repeat(colors, ntv, 0).astype(np.float32)
    offs = np.arange(n, dtype=np.int32)[:,None]*ntv
    all_f = (np.tile(tf,(n,1)) + np.repeat(offs,ntf,0)).astype(np.int32)
    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tot_v, tot_f = n*ntv, n*ntf
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {tot_v}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property float red\nproperty float green\nproperty float blue\n"
           f"element face {tot_f}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(tmp.name, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(np.hstack([all_v, all_c]).astype(np.float32).tobytes())
        buf = bytearray(tot_f*13); o = 0
        for i in range(tot_f):
            buf[o]=3; o+=1
            struct.pack_into("<iii", buf, o, int(all_f[i,0]), int(all_f[i,1]), int(all_f[i,2]))
            o+=12
        f.write(bytes(buf))
    return tmp.name

# =========================================================================== #
#  Mitsuba scene — roughplastic + 3-point lighting + ground shadow
# =========================================================================== #
def _scene(ply, res=(700,700), spp=256, cam=(2,-2,1), fov=38):
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        "sensor": {
            "type": "perspective", "fov": fov,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=cam, target=(0,0,0), up=(0,0,1)),
            "film": {"type": "hdrfilm", "width": res[0], "height": res[1],
                     "pixel_format": "rgb", "rfilter": {"type": "gaussian"}},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        # Key light — warm, upper right
        "key": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(3, -1, 5), target=(0,0,0), up=(0,0,1))
                @ mi.ScalarTransform4f.scale((1.2, 1.2, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [8.0, 7.5, 7.0]}},
        },
        # Fill light — cool, left
        "fill": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(-4, 1, 3), target=(0,0,0), up=(0,0,1))
                @ mi.ScalarTransform4f.scale((2.5, 2.5, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [2.0, 2.2, 2.8]}},
        },
        # Rim light — behind, subtle
        "rim": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(-1, 3, 2), target=(0,0,0), up=(0,0,1))
                @ mi.ScalarTransform4f.scale((1.5, 1.5, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [3.0, 3.0, 3.2]}},
        },
        # Ambient dome — very soft top fill for AO readability
        "dome": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=(0, 0, 7), target=(0,0,0), up=(0,1,0))
                @ mi.ScalarTransform4f.scale((5, 5, 1)),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": [0.6, 0.6, 0.65]}},
        },
        # Ground plane — catches shadows
        "ground": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0,0,-1.05))
                @ mi.ScalarTransform4f.scale((10, 10, 1)),
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "rgb", "value": [0.82, 0.80, 0.78]}},
        },
        # Point cloud — principled BSDF (diffuse + subtle specular = ceramic look)
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
                    "roughness": 0.45,
                    "specular": 0.3,
                    "metallic": 0.0,
                },
            },
        },
    }


def render(pts, colors, res=(700,700), spp=256, cam=(2,-2,1)):
    """Render (N,3) points with (N,3) RGB colors, return uint8 image."""
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(0); pts = pts - c
    mx = np.abs(pts).max()
    if mx > 1e-6: pts = pts * (0.8 / mx)
    # Adaptive radius
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=7)
    radius = float(0.45 * np.median(d[:, 1:]))
    radius = np.clip(radius, 0.003, 0.020)
    colors = np.asarray(colors, dtype=np.float32)
    ply = _build_ply(pts, colors, radius)
    try:
        scene = mi.load_dict(_scene(ply, res, spp, cam))
        img = np.array(mi.render(scene))
    finally:
        os.unlink(ply)
    img = np.clip(img, 0, None)
    img = img / (1.0 + img)                        # Reinhard
    img = np.power(np.clip(img, 0, 1), 1.0/2.2)   # sRGB
    return (img * 255).clip(0, 255).astype(np.uint8)


# =========================================================================== #
#  Coloring: green surface + red noise
# =========================================================================== #

def color_green_red(pred, gt, threshold_mult=2.5):
    """Green for on-surface, red for outliers.

    threshold = threshold_mult * median(NN-distance of baseline).
    Smooth sigmoid transition around threshold.
    """
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)
    dists = dists.astype(np.float32)
    median_d = np.median(dists)
    threshold = threshold_mult * median_d

    # Sigmoid blend: 0 at dist=0, 1 at dist>>threshold
    t = 1.0 / (1.0 + np.exp(-8.0 * (dists / threshold - 1.0)))
    t = t[:, None]  # (N, 1)
    return GREEN * (1.0 - t) + RED * t


# =========================================================================== #
#  Data & denoising (same as before)
# =========================================================================== #

def load_shape(name, axis_swap, npoints=2048):
    pcl_dir = os.path.join(SHAPENET_DATA, "PUNet", "pointclouds", "test", "10000_poisson")
    pc = np.loadtxt(os.path.join(pcl_dir, f"{name}.xyz"), dtype=np.float32)
    rng = np.random.RandomState(0)
    if pc.shape[0] > npoints:
        pc = pc[rng.choice(pc.shape[0], npoints, replace=False)]
    if axis_swap is not None:
        pc = pc[:, axis_swap]
    pc -= pc.mean(0)
    pc /= np.sqrt((pc**2).sum(1).max())
    return pc


def bilateral_denoise(pts, k=16, sigma_n=0.5, iters=2):
    pts = pts.copy()
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k+1)
    sigma_d = float(np.median(d[:,1:])) * 2.0
    for _ in range(iters):
        tree = cKDTree(pts)
        d, idx = tree.query(pts, k=k+1)
        nb = pts[idx[:,1:]]
        diff = nb - pts[:,None,:]
        w = np.exp(-d[:,1:]**2 / (2*sigma_d**2))
        w /= w.sum(1, keepdims=True)
        pts = pts + sigma_n * (diff * w[:,:,None]).sum(1)
    return pts


def sor_denoise(pts, k=20, std_ratio=1.0):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k+1)
    md = d[:,1:].mean(1)
    return pts[md < md.mean() + std_ratio * md.std()]


def run_denoising(trainer, clean_dict, noise_std, device):
    from sb_cover.evaluation.ddpm_sampling import ddpm_denoise
    from sb_cover.evaluation.guided_sampling import (
        ddpm_denoise_guided, ddpm_denoise_ortho_guided, GeometricQualityLoss)
    gl = GeometricQualityLoss(w_repulsion=1.0, w_projection=1.0, w_covariance=0, subsample_n=512)

    names = list(clean_dict.keys())
    batch = torch.stack([torch.from_numpy(clean_dict[n].T).float() for n in names]).to(device)
    rng = torch.Generator(device=device); rng.manual_seed(42)
    noisy = batch + noise_std * torch.randn(batch.shape, device=device, generator=rng)

    noisy_dict = {n: noisy[i].T.cpu().numpy() for i, n in enumerate(names)}
    results = {}

    for label, mtype, lam, ann in METHODS:
        print(f"  [{label}]...", end="", flush=True)
        if mtype == "baseline":
            with torch.no_grad():
                r = ddpm_denoise(trainer.model, trainer.sb_schedule, noisy,
                                 sampling_steps=10, verbose=False)
        elif mtype == "std":
            r = ddpm_denoise_guided(trainer.model, trainer.sb_schedule, noisy,
                sampling_steps=10, guidance_scale=lam, annealing=ann,
                geom_loss=gl, grad_clip=2.0, verbose=False)
        else:
            r = ddpm_denoise_ortho_guided(trainer.model, trainer.sb_schedule, noisy,
                sampling_steps=10, guidance_scale=lam, annealing=ann,
                geom_loss=gl, grad_clip=2.0, verbose=False)
        results[label] = {n: r["x_pred"][i].T.cpu().numpy() for i, n in enumerate(names)}
        print(" done")

    print("  [Bilateral]...", end="", flush=True)
    results["Bilateral"] = {n: bilateral_denoise(noisy_dict[n]) for n in names}
    print(" done")
    print("  [SOR]...", end="", flush=True)
    results["SOR"] = {n: sor_denoise(noisy_dict[n]) for n in names}
    print(" done")

    return noisy_dict, results


# =========================================================================== #
#  Figure composition
# =========================================================================== #

def _font(size=24):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()


def compose(clean_dict, noisy_dict, results, shape_info,
            res=(700,700), spp=256, save_path=None):
    names  = [s[0] for s in shape_info]
    cams   = [s[2] for s in shape_info]
    n_rows, n_cols = len(names), len(COL_LABELS)

    # Compute shared threshold per shape (based on P2P-Bridge baseline)
    thresholds = {}
    for name in names:
        gt = clean_dict[name]
        baseline = results["P2P-Bridge"][name]
        tree = cKDTree(gt)
        d, _ = tree.query(baseline, k=1)
        thresholds[name] = float(np.median(d)) * 2.5

    print(f"\nRendering {n_rows} x {n_cols} = {n_rows*n_cols} panels...")
    panels = [[None]*n_cols for _ in range(n_rows)]

    for ri, name in enumerate(names):
        gt  = clean_dict[name]
        cam = cams[ri]
        thr = thresholds[name]

        print(f"  Row {ri} ({name}): ", end="", flush=True)

        # GT — uniform gray plaster
        print("GT", end="", flush=True)
        panels[ri][0] = render(gt, np.tile(GRAY, (len(gt),1)), res, spp, cam)

        # Noisy — green/red
        print(", Noisy", end="", flush=True)
        noisy_np = noisy_dict[name]
        c = color_green_red(noisy_np, gt, threshold_mult=2.5)
        panels[ri][1] = render(noisy_np, c, res, spp, cam)

        # Methods
        for ci, label in enumerate(COL_LABELS[2:], start=2):
            print(f", {label}", end="", flush=True)
            pred = results[label][name]
            c = color_green_red(pred, gt, threshold_mult=2.5)
            panels[ri][ci] = render(pred, c, res, spp, cam)
        print()

    # Assemble
    ph, pw = panels[0][0].shape[:2]
    title_h  = 55
    label_w  = 95
    gap      = 3
    metric_h = 22

    total_w = label_w + n_cols * pw + (n_cols - 1) * gap
    total_h = title_h + n_rows * (ph + metric_h) + (n_rows - 1) * gap

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 245  # very light warm gray
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    tf, lf, mf = _font(22), _font(17), _font(13)

    # Column headers
    for ci, lab in enumerate(COL_LABELS):
        xc = label_w + ci * (pw + gap) + pw // 2
        bb = draw.textbbox((0,0), lab, font=tf)
        draw.text((xc - (bb[2]-bb[0])//2, 14), lab, fill=(40,40,40), font=tf)

    for ri, name in enumerate(names):
        gt = clean_dict[name]
        yo = title_h + ri * (ph + metric_h + gap)

        # Row label
        bb = draw.textbbox((0,0), name, font=lf)
        draw.text((8, yo + ph//2 - (bb[3]-bb[1])//2), name,
                  fill=(60,60,60), font=lf)

        for ci in range(n_cols):
            xo = label_w + ci * (pw + gap)
            pil.paste(Image.fromarray(panels[ri][ci]), (xo, yo))

            # Metric: mean NN-error + outlier%
            if ci >= 2:
                lab = COL_LABELS[ci]
                pred = results[lab][name]
                tree = cKDTree(gt)
                d, _ = tree.query(pred, k=1)
                me = float(d.mean() * 1000)
                pct = float((d > thresholds[name]).mean() * 100)
                txt = f"Err={me:.1f}  Out={pct:.0f}%"
                bb = draw.textbbox((0,0), txt, font=mf)
                draw.text((xo + pw//2 - (bb[2]-bb[0])//2, yo + ph + 2),
                          txt, fill=(80,80,80), font=mf)

    # Legend: small green/red circles
    lx = total_w - 180
    ly = total_h - 18
    draw.ellipse([lx, ly, lx+12, ly+12], fill=(int(GREEN[0]*255), int(GREEN[1]*255), int(GREEN[2]*255)))
    draw.text((lx+16, ly-2), "On-surface", fill=(80,80,80), font=mf)
    draw.ellipse([lx+100, ly, lx+112, ly+12], fill=(int(RED[0]*255), int(RED[1]*255), int(RED[2]*255)))
    draw.text((lx+116, ly-2), "Outlier", fill=(80,80,80), font=mf)

    result = np.array(pil)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pil.save(save_path, quality=95)
        print(f"\nSaved: {save_path}")
    return result


# =========================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--res", type=int, default=700)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="configs/shapenet_denoise_sb_igv.yaml")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Clean Ceramic Rendering — Green Surface / Red Noise")
    print("=" * 70)
    print(f"  sigma={args.noise}  spp={args.spp}  res={args.res}\n")

    # Load shapes
    print("Loading shapes...")
    clean_dict = {}
    for name, swap, cam in SHAPES:
        pc = load_shape(name, swap)
        clean_dict[name] = pc
        print(f"  {name}: {pc.shape}")

    # Load model
    print("\nLoading model...")
    from omegaconf import OmegaConf
    from sb_cover.training.trainer_igv import TrainerIGV
    cfg = OmegaConf.load(args.config)
    trainer = TrainerIGV(cfg, device=device)
    trainer.load_pretrained_backbone("pretrained/PVDS_PUNet/latest.pth")
    trainer.model.eval()

    print(f"\nDenoising (sigma={args.noise})...")
    noisy_dict, results = run_denoising(trainer, clean_dict, args.noise, device)

    compose(
        clean_dict, noisy_dict, results, SHAPES,
        res=(args.res, args.res), spp=args.spp,
        save_path=os.path.join(args.output_dir,
                               f"punet_clean_sigma{args.noise:.3f}.png"))
    print("Done.")


if __name__ == "__main__":
    main()
