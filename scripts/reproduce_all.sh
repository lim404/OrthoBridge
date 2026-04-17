#!/bin/bash
# Reproduce all paper tables and figures.
# Usage: bash scripts/reproduce_all.sh
#
# Prerequisites:
#   - Pretrained model at pretrained/PVDS_PUNet/latest.pth
#   - PU-Net data at data/objects/PUNet/
#   - Environment set up (bash install.sh)

set -e
cd "$(dirname "$0")/.."

CKPT="pretrained/PVDS_PUNet/latest.pth"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found at $CKPT"
    echo "Download from: https://drive.google.com/drive/folders/1hkd_gTU2EAMFJmgUzHmifviKDVunb6aK"
    exit 1
fi

echo "=============================================="
echo "  P2P-Bridge / OrthoBridge — Full Reproduction"
echo "=============================================="

# ── Table 1: Standard evaluation ──
echo ""
echo "[1/7] Standard evaluation (PU-Net)..."
python evaluate_objects.py --model_path "$CKPT" --dataset PUNet

# ── Table 2: Guidance mechanism ablation ──
echo ""
echo "[2/7] Guidance ablation (Table 2)..."
python tools/eval_guidance_ablation.py \
    --num_shapes 100 --noise_std 0.03 0.05 \
    --lambdas 1.0 5.0 10.0 20.0

# ── Figure 4: Score-as-normal validation ──
echo ""
echo "[3/7] Score-as-normal validation (Figure 4)..."
python tools/eval_score_as_normal.py \
    --num_objects 10 --sampling_steps 20 --sigmas 0.01 0.02 0.03

# ── Table 3: Schedule ablation ──
echo ""
echo "[4/7] Schedule ablation (Table 3)..."
python tools/eval_schedule_ablation.py \
    --num_shapes 100 --lambdas 0.3 3.0

# ── Table 4: Multi-seed stability ──
echo ""
echo "[5/7] Multi-seed stability (Table 4)..."
python tools/eval_multiseed.py \
    --seeds 42 123 456 789 1024 --num_shapes 100

# ── Table 5: Generalizability ──
echo ""
echo "[6/7] Generalizability (Table 5)..."
python tools/eval_generalizability.py \
    --num_shapes 100 --guidance_scale 3.0

# ── Figure 5: Metric calibration ──
echo ""
echo "[7/7] Metric calibration (Figure 5)..."
python tools/eval_metric_calibration.py \
    --num_objects 10 --noise_std 0.03

echo ""
echo "=============================================="
echo "  All experiments complete."
echo "  Results in: experiments/"
echo "  Figures in: figures/"
echo "=============================================="
