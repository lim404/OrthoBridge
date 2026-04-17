"""Training entry-point for OrthoBridge model.

Usage:
    python train_sb_igv.py --config configs/shapenet_denoise_sb_igv.yaml \
                           --data_dir /path/to/data/objects \
                           --output_dir checkpoints/sb_igv
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from sb_cover.data.punet_loader import get_punet_loaders
from sb_cover.training.trainer_igv import TrainerIGV


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train OrthoBridge model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/shapenet_denoise_sb_igv.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory in config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sb_igv",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained P2P-Bridge backbone checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sb-igv",
        help="Wandb project name",
    )
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Override data_dir if specified
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir

    # Seed
    seed = cfg.training.get("seed", 42)
    set_seed(seed)

    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))
    logger.info("Data dir: {}", cfg.data.data_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)
    if torch.cuda.is_available():
        logger.info("GPU: {}", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: {:.1f} GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Data loaders
    data_cfg = cfg.data
    logger.info("Loading PUNet data...")
    train_loader, test_loader = get_punet_loaders(
        data_dir=data_cfg.data_dir,
        patch_size=data_cfg.get("npoints", 2048),
        batch_size=cfg.training.get("batch_size", 16),
        noise_min=data_cfg.get("noise_min", 0.01),  # match pretrained PVDS_PUNet
        noise_max=data_cfg.get("noise_max", 0.02),  # match pretrained PVDS_PUNet
        num_workers=data_cfg.get("workers", 4),
        num_patches=data_cfg.get("num_patches", 200),
    )
    logger.info(
        "Train: {} batches, Test: {} batches",
        len(train_loader),
        len(test_loader),
    )

    # Trainer
    logger.info("Building trainer...")
    trainer = TrainerIGV(cfg, device=device)

    # Load pretrained backbone
    if args.pretrained is not None:
        trainer.load_pretrained_backbone(args.pretrained)

    # Resume
    start_epoch = 0
    if args.resume is not None:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info("Resumed from epoch {}", start_epoch)

    # Wandb
    wandb_logger = None
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=OmegaConf.to_container(cfg))
            wandb_logger = wandb
            logger.info("Wandb initialized: {}", wandb.run.name)
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    epochs = cfg.training.get("epochs", 200)
    save_interval = cfg.training.get("save_interval", 10)

    logger.info("Starting training for {} epochs...", epochs)

    for epoch in range(start_epoch, epochs):
        avg_loss = trainer.train_epoch(
            train_loader,
            epoch=epoch,
            wandb_logger=wandb_logger,
        )

        logger.info(
            "[Epoch {:>3d}/{:>3d}] avg_loss: {:.6f} lr: {:.2e}",
            epoch,
            epochs,
            avg_loss,
            trainer.optimizer.param_groups[0]["lr"],
        )

        if wandb_logger is not None:
            wandb_logger.log({
                "epoch": epoch,
                "epoch/avg_loss": avg_loss,
                "epoch/lr": trainer.optimizer.param_groups[0]["lr"],
            })

        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            trainer.save_checkpoint(epoch, args.output_dir)

    logger.info("Training complete.")

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
