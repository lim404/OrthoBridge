"""Schrödinger Bridge noise schedule for OrthoBridge model.

Ported from P2P-Bridge/models/p2pb.py with modifications for the OrthoBridge framework.
Implements symmetric sqrt-linear beta schedule, forward process q_sample,
ground truth computation, and reverse posterior for DDPM sampling.
"""

from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def make_beta_schedule(
    n_timestep: int = 1000,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
) -> np.ndarray:
    """Create sqrt-linear beta schedule.

    Args:
        n_timestep: Number of diffusion timesteps.
        linear_start: Starting beta value (before sqrt scaling).
        linear_end: Ending beta value (before sqrt scaling).

    Returns:
        Beta schedule array of shape (n_timestep,).
    """
    scale = 1000 / n_timestep
    linear_start *= scale
    linear_end *= scale
    betas = torch.linspace(
        linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
    ) ** 2
    return betas.numpy()


def compute_gaussian_product_coef(
    sigma1: np.ndarray, sigma2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coefficients for Gaussian product.

    Given two Gaussians N(x; mu1, sigma1^2) and N(x; mu2, sigma2^2),
    their product is proportional to N(x; mu, var) where:
        mu = (sigma2^2 * mu1 + sigma1^2 * mu2) / (sigma1^2 + sigma2^2)
        var = (sigma1^2 * sigma2^2) / (sigma1^2 + sigma2^2)

    Args:
        sigma1: Standard deviations of first Gaussian.
        sigma2: Standard deviations of second Gaussian.

    Returns:
        Tuple of (coef1, coef2, var) where coef1 weights mu1, coef2 weights mu2.
    """
    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom
    return coef1, coef2, var


def space_indices(num_steps: int, count: int) -> List[int]:
    """Generate evenly spaced indices over a range of steps.

    Args:
        num_steps: Total number of steps.
        count: Number of indices to generate.

    Returns:
        List of evenly spaced integer indices.
    """
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def unsqueeze_xdim(z: Tensor, xdim: Tuple[int, ...]) -> Tensor:
    """Unsqueeze tensor to match spatial dimensions.

    Args:
        z: Tensor to unsqueeze.
        xdim: Target spatial dimensions (excluding batch).

    Returns:
        Tensor with singleton dimensions appended.
    """
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


class SBSchedule(nn.Module):
    """Schrödinger Bridge noise schedule.

    Implements the forward process, ground truth computation, and reverse
    posterior for a symmetric Schrödinger Bridge diffusion process.

    Args:
        n_timestep: Number of diffusion timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        symmetric: Whether to use symmetric beta schedule.
        objective: Prediction objective ('pred_x0' or 'pred_noise').
        ot_ode: If True, use OT-ODE (deterministic); if False, add stochastic noise.
        device: Device to place tensors on.
    """

    def __init__(
        self,
        n_timestep: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        symmetric: bool = True,
        objective: str = "pred_x0",
        ot_ode: bool = False,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_timestep = n_timestep
        self.objective = objective
        self.ot_ode = ot_ode

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build beta schedule
        betas = make_beta_schedule(
            n_timestep=n_timestep,
            linear_start=beta_start,
            linear_end=beta_end,
        )

        if symmetric:
            betas = np.concatenate([
                betas[:n_timestep // 2],
                np.flip(betas[:n_timestep // 2]),
            ])

        # Compute forward and backward standard deviations
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))

        # Compute Gaussian product coefficients for SB interpolation
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # Noise levels for time embedding (matches P2P-Bridge convention)
        noise_levels = torch.linspace(
            1e-4, 1.0, n_timestep, dtype=torch.float32
        ) * n_timestep

        # Compute SNR (signal-to-noise ratio) at each timestep
        # SNR(t) = mu_x0(t)^2 / std_sb(t)^2  (how much signal vs noise)
        # High SNR = clean, Low SNR = noisy
        snr = np.where(var > 1e-12, mu_x0 ** 2 / var, 1e6)

        # Register all as buffers
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("std_fwd", to_torch(std_fwd))
        self.register_buffer("std_bwd", to_torch(std_bwd))
        self.register_buffer("mu_x0", to_torch(mu_x0))
        self.register_buffer("mu_x1", to_torch(mu_x1))
        self.register_buffer("std_sb", to_torch(std_sb))
        self.register_buffer("noise_levels", noise_levels)
        self.register_buffer("snr", to_torch(snr))

    def snr_weight(
        self,
        step: Tensor,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        schedule: str = "sigmoid",
        sharpness: float = 1.0,
        midpoint_snr: float = 1.0,
    ) -> Tensor:
        """Compute SNR-adaptive weight for IGV losses at given timesteps.

        Returns a per-sample weight in [clamp_min, clamp_max]:
        - Near 0 when SNR is low (noisy, high t) → don't enforce geometry
        - Near 1 when SNR is high (clean, low t) → enforce geometry fully

        Args:
            step: Timestep indices (B,).
            clamp_min: Minimum weight value.
            clamp_max: Maximum weight value.
            schedule: Weight schedule type ('sigmoid', 'linear', 'cosine').
            sharpness: Controls sigmoid steepness (higher = sharper transition).
            midpoint_snr: SNR value at which weight = 0.5 (for sigmoid).

        Returns:
            Per-sample weights (B,).
        """
        snr_t = self.snr[step]  # (B,)

        if schedule == "sigmoid":
            # σ(a * (log(SNR) - log(midpoint)))
            # High SNR → weight ≈ 1, Low SNR → weight ≈ 0
            log_snr = torch.log(snr_t.clamp(min=1e-8))
            log_mid = torch.log(torch.tensor(midpoint_snr, device=step.device))
            weight = torch.sigmoid(sharpness * (log_snr - log_mid))

        elif schedule == "linear":
            # Normalize SNR to [0, 1] using max SNR
            snr_max = self.snr.max()
            weight = (snr_t / snr_max.clamp(min=1e-8)).clamp(0.0, 1.0)

        elif schedule == "cosine":
            # Map normalized timestep to cosine curve
            # t=0 → weight=1, t=T → weight=0
            t_normalized = step.float() / max(self.n_timestep - 1, 1)
            weight = 0.5 * (1.0 + torch.cos(torch.pi * t_normalized))

        else:
            raise ValueError(f"Unknown SNR weight schedule: {schedule}")

        return weight.clamp(clamp_min, clamp_max)

    def get_std_fwd(self, step: Tensor, xdim: Tuple[int, ...] = None) -> Tensor:
        """Get forward std at given timesteps.

        Args:
            step: Timestep indices (B,).
            xdim: Spatial dimensions for unsqueezing.

        Returns:
            Forward standard deviation tensor.
        """
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Forward process: sample xt given x0 (clean) and x1 (noisy).

        xt = mu_x0[t] * x0 + mu_x1[t] * x1 + std_sb[t] * noise

        Args:
            step: Timestep indices (B,).
            x0: Clean samples (B, ...).
            x1: Noisy samples (B, ...).

        Returns:
            Interpolated samples xt (B, ...).
        """
        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1

        if not self.ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)

        return xt.detach()

    def compute_gt(self, step: Tensor, x0: Tensor, xt: Tensor) -> Tensor:
        """Compute ground truth target for the given objective.

        Args:
            step: Timestep indices (B,).
            x0: Clean samples (B, ...).
            xt: Interpolated samples (B, ...).

        Returns:
            Ground truth target tensor.
        """
        if self.objective == "pred_noise":
            std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
            gt = (xt - x0) / std_fwd
            return gt.detach()
        elif self.objective == "pred_x0":
            return x0.detach()
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def compute_pred_x0(self, step: Tensor, xt: Tensor, net_out: Tensor) -> Tensor:
        """Compute predicted x0 from network output.

        Args:
            step: Timestep indices (B,).
            xt: Noisy input (B, ...).
            net_out: Network prediction (B, ...).

        Returns:
            Predicted clean sample x0.
        """
        if self.objective == "pred_x0":
            return net_out
        elif self.objective == "pred_noise":
            std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
            return xt - std_fwd * net_out
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def p_posterior(
        self, nprev: int, n: int, x_n: Tensor, x0: Tensor
    ) -> Tensor:
        """Compute reverse step posterior p(x_{t-1} | x_t, x_0).

        Args:
            nprev: Previous timestep index (t-1).
            n: Current timestep index (t).
            x_n: Current latent state x_t (B, ...).
            x0: Predicted clean sample (B, ...).

        Returns:
            Previous latent state x_{t-1} (B, ...).
        """
        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n ** 2 - std_nprev ** 2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(
            std_nprev.cpu().numpy(), std_delta.cpu().numpy()
        )

        mu_x0 = torch.tensor(mu_x0, dtype=x_n.dtype, device=x_n.device)
        mu_xn = torch.tensor(mu_xn, dtype=x_n.dtype, device=x_n.device)
        var = torch.tensor(var, dtype=x_n.dtype, device=x_n.device)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not self.ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def space_indices(self, count: int) -> List[int]:
        """Get evenly spaced sampling step indices.

        Args:
            count: Number of sampling steps.

        Returns:
            List of step indices.
        """
        return space_indices(self.n_timestep, count)
