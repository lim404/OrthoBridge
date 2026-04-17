<p align="center">
  <h1 align="center">P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising</h1>
  <p align="center">
    <a href="https://matvogel.github.io">Mathias Vogel</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=ml3laqEAAAAJ">Keisuke Tateno</a><sup>2</sup>,
    <a href="https://inf.ethz.ch/people/person-detail.pollefeys.html">Marc Pollefeys</a><sup>1,3</sup>,
    <a href="https://federicotombari.github.io/">Federico Tombari</a><sup>2,4</sup>,
    <a href="https://scholar.google.com/citations?user=eQ0om98AAAAJ">Marie-Julie Rakotosaona</a><sup>*2</sup>,
    <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>*1,2</sup>
    <br>
    <sup>1</sup>ETH Zurich,
    <sup>2</sup>Google,
    <sup>3</sup>Microsoft,
    <sup>4</sup>TUM
    <br>
    <sup>*</sup>Equal Contribution
  </p>
  <h3 align="center">
    <a href="./assets/P2P-Bridge.pdf">Paper</a> |
    <a href="https://drive.google.com/drive/folders/1hkd_gTU2EAMFJmgUzHmifviKDVunb6aK?usp=sharing">Pretrained Models</a>
  </h3>
</p>

<p align="center">
  <img src="./assets/overview.png" width="100%">
</p>

**P2P-Bridge** introduces a novel framework for 3D point cloud denoising by adapting Diffusion Schr&ouml;dinger Bridges to learn an optimal transport plan between noisy and clean point sets. We further propose **SB-IGV** (Schr&ouml;dinger Bridge with Integral Geometry Validation), which enhances P2P-Bridge with:

- **Orthogonal Gradient Projection Guidance** &mdash; training-free geometric guidance that projects quality gradients orthogonal to the denoising score direction, improving surface uniformity (VD) without degrading silhouette consistency (IGSD).
- **Novel evaluation metrics** &mdash; Valuation Difference (VD) and Integral Geometry Signature Distance (IGSD) for detecting morphological shrinkage and topological breaks that standard CD/EMD miss.

## Results Summary

### Guidance Mechanism Comparison (same backbone, &sigma;=0.03)

| Method | &lambda; | CD &darr; | VD &darr; | IGSD &darr; |
|---|---|---|---|---|
| Baseline (no guidance) | &mdash; | 50.30 | 0.0248 | 0.000091 |
| Standard guidance | 10 | 92.85 (+85%) | 0.2651 (+969%) | 0.003402 |
| Clipped/Normalized | 10 | 66.81 (+33%) | 0.1320 (+432%) | 0.000660 |
| **Orthogonal (ours)** | **10** | **52.85 (+5%)** | **0.0329 (+33%)** | **0.000110** |

At &lambda;=10, orthogonal guidance is **8&times; better on VD** and **31&times; better on IGSD** than standard guidance.

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.1+

### Setup

```bash
# Create environment
conda create -n p2pb python=3.10
conda activate p2pb

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia --yes

# Install PyTorch3D and TorchCluster
# See: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# See: https://github.com/rusty1s/pytorch_cluster

# Install dependencies and compile CUDA extensions
bash install.sh
```

## Data Preparation

### Object Datasets (PU-Net / PC-Net)

Download from [ScoreDenoise](https://github.com/luost26/score-denoise) and extract:

```
data/objects/
├── examples/
├── PCNet/
└── PUNet/
```

### Indoor Scenes (ScanNet++ / ARKitScenes)

Follow the instructions in [`data/readme.md`](data/readme.md).

## Pretrained Models

Download from [Google Drive](https://drive.google.com/drive/folders/1hkd_gTU2EAMFJmgUzHmifviKDVunb6aK?usp=sharing) and extract:

```
pretrained/
├── PVDS_PUNet/
│   └── latest.pth
└── PVDL_ARK_XYZ/
    └── step_100000.pth
```

## Training

```bash
# PU-Net denoising (small model)
python train.py --config configs/PVDS_PUNet.yaml --save_dir checkpoints/punet

# SB-IGV training (on pretrained P2P-Bridge backbone)
python train_sb_igv.py --config configs/shapenet_denoise_sb_igv.yaml
```

Training uses [Weights & Biases](https://wandb.ai) for logging. Run `wandb disabled` to turn it off.

## Evaluation

### Standard Evaluation

```bash
# PU-Net test set
python evaluate_objects.py --model_path pretrained/PVDS_PUNet/latest.pth --dataset PUNet

# PC-Net test set
python evaluate_objects.py --model_path pretrained/PVDS_PUNet/latest.pth --dataset PCNet

# Indoor scenes (ScanNet++)
bash scripts/denoise_snpp.sh <PATH_TO_DATA>
python evaluate_rooms.py --data_root <PATH_TO_DATA> --dataset snpp
```

### Ablation & Analysis Tools

All ablation scripts are in `tools/`. Run from the project root:

```bash
# Guidance mechanism comparison (Table 2)
python tools/eval_guidance_ablation.py --num_shapes 100 --noise_std 0.03 0.05

# Score-as-normal validation (Figure 4)
python tools/eval_score_as_normal.py --num_objects 10 --sampling_steps 20

# Schedule ablation (Table 3)
python tools/eval_schedule_ablation.py --lambdas 0.3 3.0

# Multi-seed stability (Table 4)
python tools/eval_multiseed.py --seeds 42 123 456 789 1024

# Cross-backbone & cross-objective generalizability (Table 5)
python tools/eval_generalizability.py --num_shapes 100

# Metric external calibration (Figure 5)
python tools/eval_metric_calibration.py --num_objects 10
```

### One-Command Reproduction

```bash
# Reproduce all paper tables and figures
bash scripts/reproduce_all.sh
```

## Denoise Your Own Data

<p align="center">
  <img src="./assets/room-denoise.gif" width="60%">
</p>

```bash
# Objects (XYZ format)
python denoise_object.py --data_path input.xyz --save_path output.xyz \
    --model_path pretrained/PVDS_PUNet/latest.pth

# Indoor scenes
python denoise_room.py --room_path <ROOM_PATH> --model_path <MODEL_PATH> \
    --out_path <OUTPUT_PATH>
```

## Project Structure

```
P2P-Bridge/
├── configs/                    # Training configurations
├── models/                     # P2P-Bridge backbone (PVCNN, UNet)
├── sb_cover/                   # SB-IGV module
│   ├── data/                   #   Data loading (PU-Net patches)
│   ├── evaluation/             #   Sampling & guided inference
│   │   ├── ddpm_sampling.py    #     DDPM reverse process
│   │   └── guided_sampling.py  #     Orthogonal guidance (core contribution)
│   ├── losses/                 #   SB schedule & IGV losses
│   ├── models/                 #   Bridge model & decoder
│   └── training/               #   Training loop
├── metrics/                    # Evaluation metrics (CD, EMD, VD, IGSD)
│   └── geometric_metrics.py    #   Novel VD & IGSD metrics
├── tools/                      # Ablation & analysis scripts
├── third_party/                # External dependencies (OpenPoints)
├── scripts/                    # Shell scripts for reproduction
├── train.py                    # Training entry point
├── train_sb_igv.py             # SB-IGV training
├── evaluate_objects.py         # Object evaluation
├── evaluate_rooms.py           # Scene evaluation
├── denoise_object.py           # Single-object inference
└── denoise_room.py             # Scene inference
```

### Key Files

| File | Description |
|---|---|
| `sb_cover/evaluation/guided_sampling.py` | Orthogonal gradient projection guidance |
| `sb_cover/losses/sb_interpolation.py` | Schr&ouml;dinger Bridge noise schedule |
| `sb_cover/models/bridge_model.py` | Bridge flow model with FPS + decoder |
| `metrics/geometric_metrics.py` | VD and IGSD metric implementations |
| `tools/eval_guidance_ablation.py` | Guidance mechanism comparison |
| `tools/eval_score_as_normal.py` | Score-as-normal validation |

## Citation

```bibtex
@inproceedings{vogel2024p2pbridge,
    title     = {P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising},
    author    = {Mathias Vogel and Keisuke Tateno and Marc Pollefeys and
                 Federico Tombari and Marie-Julie Rakotosaona and Francis Engelmann},
    year      = {2024},
    booktitle = {European Conference on Computer Vision (ECCV)},
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

This codebase builds upon [P2P-Bridge](https://github.com/matvogel/P2P-Bridge) (ECCV 2024). We thank the original authors for their excellent work and open-source release.
