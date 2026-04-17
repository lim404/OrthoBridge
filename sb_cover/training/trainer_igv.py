"""IGV Trainer: Training loop for OrthoBridge model.

Supports both legacy flow matching and Schrödinger Bridge noise schedule.
When use_sb_schedule=True, uses discrete timesteps, SB forward process,
and point-level MSE as primary loss with IGV losses on dense predictions.
"""

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
from ema_pytorch import EMA
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.amp import autocast

import pointnet2_batch_cuda

from sb_cover.losses.combined_igv import CombinedIGVLoss
from sb_cover.losses.sb_interpolation import SBSchedule
from sb_cover.models.bridge_model import BridgeFlowModel
from models.train_utils import getGradNorm, to_cuda
from models.unet_pvc import PVCNN2Unet


class TrainerIGV:
    """Training loop for OrthoBridge model.

    Args:
        cfg: Full configuration dict.
        device: CUDA device.
    """

    def __init__(self, cfg: DictConfig, device: torch.device = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_cfg = cfg.get("loss", {})
        self.use_sb_schedule = loss_cfg.get("use_sb_schedule", False)

        # Build backbone
        backbone = PVCNN2Unet(cfg)
        logger.info(
            "Backbone params (M): {:.2f}",
            sum(p.numel() for p in backbone.parameters() if p.requires_grad) / 1e6,
        )

        # Build bridge model
        self.model = BridgeFlowModel(backbone, cfg).to(self.device)

        # Set SB objective on bridge model
        if self.use_sb_schedule:
            sb_cfg = loss_cfg.get("sb_schedule", {})
            self.model.sb_objective = sb_cfg.get("objective", "pred_x0")

            # Instantiate SB schedule
            self.sb_schedule = SBSchedule(
                n_timestep=sb_cfg.get("n_timestep", 1000),
                beta_start=sb_cfg.get("beta_start", 1e-4),
                beta_end=sb_cfg.get("beta_end", 0.02),
                symmetric=sb_cfg.get("symmetric", True),
                objective=sb_cfg.get("objective", "pred_x0"),
                ot_ode=sb_cfg.get("ot_ode", False),
                device=self.device,
            ).to(self.device)

            # Set std_fwd on model for pred_noise mode
            self.model.set_std_fwd(self.sb_schedule.std_fwd)
        else:
            self.sb_schedule = None

        # EMA
        model_cfg = cfg.get("model", {})
        if model_cfg.get("ema", False):
            ema_cfg = model_cfg.get("EMA", {})
            self.ema = EMA(self.model, beta=ema_cfg.get("decay", 0.999))
        else:
            self.ema = None

        # Combined IGV loss
        self.criterion = CombinedIGVLoss(cfg)

        # Optimizer
        train_cfg = cfg.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.get("lr", 3e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-5),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
        )

        # Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "constant")
        if sched_type == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, sched_cfg.get("lr_gamma", 0.999)
            )
        elif sched_type == "CosineAnnealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg.get("epochs", 200)
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0
            )

        # AMP — use bfloat16 (float16 causes NaN with large backbone)
        self.use_amp = train_cfg.get("amp", True)
        self.amp_dtype = torch.bfloat16

        # Grad clipping
        grad_clip_cfg = train_cfg.get("grad_clip", {})
        self.grad_clip_enabled = grad_clip_cfg.get("enabled", True)
        self.grad_clip_value = grad_clip_cfg.get("value", 1.0)

        # Training params
        self.epochs = train_cfg.get("epochs", 200)
        self.batch_size = train_cfg.get("batch_size", 32) if "batch_size" in train_cfg else train_cfg.get("bs", 32)
        self.log_interval = train_cfg.get("log_interval", 10)
        self.save_interval = train_cfg.get("save_interval", 10)
        self.accumulation_steps = train_cfg.get("accumulation_steps", 1)

        # Skip decoder during training: saves ~35% compute when decoder
        # output is not used at inference (use coarse backbone output instead)
        self.skip_decoder = train_cfg.get("skip_decoder", False)
        self.full_mse_weight = train_cfg.get("full_mse_weight", 1.0)

        # Mask ratio: 0.0 for denoising (disable MAE masking)
        self.mask_ratio = model_cfg.get("mask_ratio", 0.0)

        # Number of centers
        self.num_centers = model_cfg.get("num_centers", 128)

    def _fps_subsample(self, points: Tensor, num_centers: int) -> Tensor:
        """Farthest Point Sampling using CUDA kernel.

        Args:
            points: (B, 3, N).
            num_centers: Number of centers.

        Returns:
            Center indices (B, num_centers) as LongTensor.
        """
        B, C, N = points.shape
        pts = points.float().transpose(1, 2).contiguous()  # (B, N, 3), ensure fp32
        output = torch.cuda.IntTensor(B, num_centers)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        pointnet2_batch_cuda.furthest_point_sampling_wrapper(
            B, N, num_centers, pts, temp, output
        )
        return output.long()

    def _gather_by_idx(self, points: Tensor, idx: Tensor) -> Tensor:
        """Gather points by FPS indices.

        Args:
            points: (B, 3, N).
            idx: (B, M).

        Returns:
            (B, 3, M).
        """
        idx_exp = idx.unsqueeze(1).expand(-1, points.shape[1], -1)
        return torch.gather(points, 2, idx_exp)

    def load_pretrained_backbone(self, ckpt_path: str):
        """Load pretrained P2P-Bridge backbone weights into the model.

        Strips the 'model.module.' prefix from checkpoint keys and loads
        only the backbone parameters. Decoder parameters are initialized fresh.

        Args:
            ckpt_path: Path to pretrained checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model_state = ckpt.get("model_state", ckpt)

        # Strip 'model.module.' prefix from pretrained keys
        backbone_state = {}
        for k, v in model_state.items():
            if k.startswith("model.module."):
                stripped = k[len("model.module."):]
                backbone_state[stripped] = v

        if not backbone_state:
            logger.warning("No 'model.module.*' keys found in checkpoint, trying raw keys")
            backbone_state = model_state

        info = self.model.backbone.load_state_dict(backbone_state, strict=False)
        logger.info(
            "Loaded pretrained backbone from {}: {} keys, missing={}, unexpected={}",
            ckpt_path,
            len(backbone_state),
            len(info.missing_keys),
            len(info.unexpected_keys),
        )
        if info.missing_keys:
            logger.warning("Missing keys: {}", info.missing_keys[:10])
        if info.unexpected_keys:
            logger.warning("Unexpected keys: {}", info.unexpected_keys[:10])

    def _bridge_step(
        self,
        x_clean: Tensor,
        x_noisy: Tensor,
        epoch: int,
    ) -> Dict[str, Tensor]:
        """Single training step.

        The backbone (PVCNN2Unet) processes the full N-point cloud and returns
        coarse predictions. The PointFlowDecoder refines these using FPS centers.
        The model returns center_idx, pred_x0_coarse, dense_pred, and velocity.

        Args:
            x_clean: Clean point cloud (B, 3, N).
            x_noisy: Noisy point cloud (B, 3, N).
            epoch: Current epoch.

        Returns:
            Dictionary of losses.
        """
        B, C, N = x_clean.shape

        if self.use_sb_schedule and self.sb_schedule is not None:
            # --- SB Schedule Mode ---
            T = self.sb_schedule.n_timestep

            # Sample discrete timesteps
            steps = torch.randint(0, T, (B,), device=self.device)

            # Forward process on FULL point cloud
            xt = self.sb_schedule.q_sample(
                steps, x_clean, x_noisy
            )  # (B, 3, N)

            # Get noise levels for time embedding
            noise_levels = self.sb_schedule.noise_levels[steps].detach()

            # Forward pass: backbone + FPS + optional decoder
            result = self.model(
                x=xt,
                t=None,
                noise_level=noise_levels,
                noisy_input=x_noisy,
                steps=steps,
                skip_decoder=self.skip_decoder,
            )

            # Use model's FPS indices for consistent center extraction
            center_idx = result["center_idx"]  # (B, M)

            # Extract GT centers using the same indices
            clean_centers = self._gather_by_idx(x_clean, center_idx)  # (B, 3, M)

            # Center-level predictions from coarse backbone output
            coarse_pred = result["pred_x0_coarse"]  # (B, 3, N)
            pred_centers = self._gather_by_idx(coarse_pred, center_idx)  # (B, 3, M)

            # Ground truth for center-level velocity loss
            xt_centers = self._gather_by_idx(xt, center_idx)
            gt_centers = self.sb_schedule.compute_gt(
                steps, clean_centers, xt_centers
            )

            # Dense prediction from decoder (or None if skipped)
            dense_pred = result["dense_pred"]

            # Compute losses — when skip_decoder, pass dense_pred=None to
            # skip score matching + chamfer (expensive O(N²) on full cloud).
            # IGV is applied to coarse output separately below.
            losses = self.criterion(
                pred_centers=pred_centers,
                gt_centers=clean_centers,
                target_points=x_clean,
                center_velocity=self._gather_by_idx(result["velocity"], center_idx),
                center_gt=gt_centers,
                dense_pred=dense_pred,  # None when skip_decoder
                epoch=epoch,
                steps=steps,
                sb_schedule=self.sb_schedule,
            )

            # When decoder is skipped, add full-cloud MSE + IGV on coarse output.
            if self.skip_decoder and dense_pred is None:
                # --- Full-cloud MSE: denoising supervision on ALL N points ---
                # Without this, only 128 FPS centers get denoising supervision
                # via center_flow, leaving 93.75% of backbone output unconstrained.
                gt_full = self.sb_schedule.compute_gt(steps, x_clean, xt)  # (B, 3, N)
                l_full_mse = nn.functional.mse_loss(result["velocity"], gt_full)
                losses["full_mse"] = l_full_mse
                losses["total"] = losses["total"] + self.full_mse_weight * l_full_mse

                # --- IGV on coarse backbone output ---
                coarse_bn3 = coarse_pred.transpose(1, 2)  # (B, N, 3)
                target_bn3 = x_clean.transpose(1, 2)      # (B, N, 3)
                curr_weights = self.criterion.curriculum.get_weights(epoch)
                mc_coarse = self.criterion.manifold_correction_dense(
                    pred_points=coarse_bn3,
                    target_points=target_bn3,
                    steps=steps,
                    sb_schedule=self.sb_schedule,
                    epoch_weight_ig=curr_weights["ig_projection"],
                    epoch_weight_val=curr_weights["valuation"],
                )
                losses["ig_dense"] = mc_coarse["ig_loss"]
                losses["val_dense"] = mc_coarse["val_loss"]
                losses["snr_weight"] = mc_coarse["snr_weight_mean"]
                losses["total"] = losses["total"] + mc_coarse["total"]

        else:
            # --- Legacy Flow Matching Mode ---
            t = torch.rand(B, device=self.device)

            # Linear interpolation on full cloud
            t_expand = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            xt = (1 - t_expand) * x_clean + t_expand * x_noisy

            # Forward pass: backbone + FPS + optional decoder
            result = self.model(x=xt, t=t, noisy_input=x_noisy,
                                skip_decoder=self.skip_decoder)

            # Use model's FPS indices
            center_idx = result["center_idx"]

            clean_centers = self._gather_by_idx(x_clean, center_idx)
            coarse_pred = result["pred_x0_coarse"]
            pred_centers = self._gather_by_idx(coarse_pred, center_idx)

            # GT velocity for centers
            noisy_centers = self._gather_by_idx(x_noisy, center_idx)
            gt_velocity = noisy_centers - clean_centers

            dense_pred = result["dense_pred"]

            losses = self.criterion(
                pred_centers=pred_centers,
                gt_centers=clean_centers,
                target_points=x_clean,
                center_velocity=self._gather_by_idx(result["velocity"], center_idx),
                center_gt=gt_velocity,
                dense_pred=dense_pred,  # None when skip_decoder
                epoch=epoch,
                steps=None,
                sb_schedule=None,
            )

            if self.skip_decoder and dense_pred is None:
                # Full-cloud MSE for legacy mode
                gt_velocity = x_noisy - x_clean
                l_full_mse = nn.functional.mse_loss(result["velocity"], gt_velocity)
                losses["full_mse"] = l_full_mse
                losses["total"] = losses["total"] + l_full_mse

                coarse_bn3 = coarse_pred.transpose(1, 2)
                target_bn3 = x_clean.transpose(1, 2)
                curr_weights = self.criterion.curriculum.get_weights(epoch)
                mc_coarse = self.criterion.manifold_correction_dense(
                    pred_points=coarse_bn3,
                    target_points=target_bn3,
                    steps=None,
                    sb_schedule=None,
                    epoch_weight_ig=curr_weights["ig_projection"],
                    epoch_weight_val=curr_weights["valuation"],
                )
                losses["ig_dense"] = mc_coarse["ig_loss"]
                losses["val_dense"] = mc_coarse["val_loss"]
                losses["total"] = losses["total"] + mc_coarse["total"]

        return losses

    def train_epoch(
        self,
        train_loader,
        epoch: int,
        wandb_logger=None,
        align_fn=None,
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
            wandb_logger: Optional wandb module for logging.
            align_fn: Optional alignment function (for PUNet dataset).

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = to_cuda(batch, self.device)

            # Extract clean and noisy points from batch
            x_clean, x_noisy = self._extract_batch(batch, align_fn)

            self.optimizer.zero_grad()

            loss_accum = torch.tensor(0.0, device=self.device)

            for accum_iter in range(self.accumulation_steps):
                with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    losses = self._bridge_step(x_clean, x_noisy, epoch)
                    loss = losses["total"] / self.accumulation_steps

                loss.backward()
                loss_accum += loss.detach()

            if self.grad_clip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_value
                )

            self.optimizer.step()

            if self.ema is not None:
                self.ema.update()

            total_loss += loss_accum.item()
            n_batches += 1

            global_step = epoch * len(train_loader) + batch_idx

            if batch_idx % self.log_interval == 0:
                pNorm, gNorm = getGradNorm(self.model)
                # Build loss component string
                loss_parts = []
                for k, v in losses.items():
                    if k != "total" and isinstance(v, Tensor) and v.item() > 0:
                        loss_parts.append(f"{k}={v.item():.4f}")
                parts_str = " ".join(loss_parts)
                logger.info(
                    "[Epoch {:>3d} Batch {:>4d}/{:>4d}] loss: {:.6f} "
                    "pNorm: {:.2f} gNorm: {:.4f} | {}",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss_accum.item(),
                    pNorm,
                    gNorm,
                    parts_str,
                )

                if wandb_logger is not None:
                    log_dict = {
                        "train/loss": loss_accum.item(),
                        "train/pNorm": pNorm,
                        "train/gNorm": gNorm,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    }
                    # Log individual loss components
                    for k, v in losses.items():
                        if k != "total" and isinstance(v, Tensor):
                            log_dict[f"train/{k}"] = v.item()
                    wandb_logger.log(log_dict, step=global_step)

        self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def _extract_batch(self, batch: Dict, align_fn=None):
        """Extract clean and noisy points from a data batch.

        Args:
            batch: Data batch dictionary.
            align_fn: Optional alignment function.

        Returns:
            Tuple of (x_clean, x_noisy) each of shape (B, 3, N).
        """
        dataset = self.cfg.data.get("dataset", "PUNet")

        if dataset == "PUNet":
            x_clean = batch["clean_points"].squeeze()
            x_noisy = batch["noisy_points"].squeeze()
        else:
            x_clean = batch["clean_points"].transpose(1, 2)
            x_noisy = batch.get("noisy_points")
            if x_noisy is not None:
                x_noisy = x_noisy.transpose(1, 2) if x_noisy.shape[1] != 3 else x_noisy

        # Ensure (B, 3, N) format
        if x_clean.dim() == 2:
            x_clean = x_clean.unsqueeze(0)
        if x_clean.shape[1] > x_clean.shape[2]:
            x_clean = x_clean.transpose(1, 2)

        if x_noisy is not None:
            if x_noisy.dim() == 2:
                x_noisy = x_noisy.unsqueeze(0)
            if x_noisy.shape[1] > x_noisy.shape[2]:
                x_noisy = x_noisy.transpose(1, 2)
        else:
            # If no noisy points, add noise to clean
            noise_std = self.cfg.data.get("noise_std", 0.05)
            x_noisy = x_clean + noise_std * torch.randn_like(x_clean)

        # Alignment for PUNet
        if align_fn is not None:
            x_clean = align_fn(x_noisy, x_clean)

        return x_clean, x_noisy

    def save_checkpoint(self, epoch: int, output_dir: str):
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            output_dir: Directory to save checkpoint.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_dict = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        if self.ema is not None:
            save_dict["ema_state"] = self.ema.state_dict()
        path = os.path.join(output_dir, f"epoch_{epoch}.pth")
        torch.save(save_dict, path)
        logger.info("Saved checkpoint to {}", path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.ema is not None and "ema_state" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state"])
        logger.info("Loaded checkpoint from {}", path)
        return ckpt.get("epoch", 0) + 1  # resume from NEXT epoch
