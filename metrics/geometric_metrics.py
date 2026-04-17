"""Geometry-aware evaluation metrics for point cloud denoising.

Novel metrics designed to detect morphological shrinkage and silhouette
degradation that standard point-wise metrics (CD, EMD) miss.

Metrics
-------
ValuationDifference (VD)
    Multi-threshold volumetric fidelity via KDE density fields.
    Quantifies global shrinkage by comparing occupied-voxel counts.

IntegralGeometrySignatureDistance (IGSD)
    Sliced Wasserstein Distance on spherical projections.
    Penalizes topological breaks and silhouette inconsistency.

References
----------
Section 4.2 of the OrthoBridge paper (ECCV 2024).
"""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "ValuationDifference",
    "IntegralGeometrySignatureDistance",
    "compute_vd",
    "compute_igsd",
]


# =========================================================================== #
#  Valuation Difference (VD)
# =========================================================================== #

class ValuationDifference:
    r"""Valuation Difference: multi-threshold volumetric fidelity metric.

    VD measures the relative error in spatial mass distribution between a
    predicted point cloud and the ground truth.  It voxelizes
    :math:`[-1,1]^3` into a :math:`G^3` grid, estimates continuous density
    via Kernel Density Estimation, and compares occupied-voxel counts across
    multiple density thresholds.

    .. math::

        \rho(x; X) = \sum_{p \in X}
            \exp\!\Bigl(-\frac{\|x - p\|^2}{2\sigma^2}\Bigr)

    .. math::

        V_\tau(X) = \sum_{x \in \mathrm{Grid}}
            \mathbb{I}\bigl(\rho(x; X) > \tau\bigr)

    .. math::

        \mathrm{VD}(X_{\text{pred}}, X_{\text{gt}})
        = \frac{1}{|\mathcal{T}|}
          \sum_{\tau \in \mathcal{T}}
          \frac{|V_\tau(X_{\text{pred}}) - V_\tau(X_{\text{gt}})|}{
                V_\tau(X_{\text{gt}}) + \epsilon}

    Lower VD = better volumetric fidelity / less shrinkage.

    Parameters
    ----------
    grid_resolution : int
        Number of voxels per axis (G).  Total grid points = G^3.
    sigma : float
        KDE bandwidth controlling density smoothness.
    thresholds : sequence of float
        Density thresholds :math:`\mathcal{T}` for volume counting.
        Using diverse thresholds ensures robustness across object scales.
    eps : float
        Small constant to prevent division by zero.
    chunk_size : int
        Grid points processed per chunk to bound GPU memory.
    """

    def __init__(
        self,
        grid_resolution: int = 64,
        sigma: float = 0.05,
        thresholds: Sequence[float] = (0.1, 0.3, 0.5, 1.0, 2.0),
        eps: float = 1e-6,
        chunk_size: int = 4096,
    ):
        self.G = grid_resolution
        self.sigma = sigma
        self.thresholds = tuple(thresholds)
        self.eps = eps
        self.chunk_size = chunk_size

        # Pre-computed grid (lazily built per device)
        self._grid_cache: dict = {}

    def _get_grid(self, device: torch.device) -> Tensor:
        """Return the voxel-center grid (G^3, 3), cached per device."""
        key = str(device)
        if key not in self._grid_cache:
            lin = torch.linspace(-1.0, 1.0, self.G, device=device)
            gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
            self._grid_cache[key] = torch.stack(
                [gx, gy, gz], dim=-1
            ).reshape(-1, 3)
        return self._grid_cache[key]

    def _kde_density(self, points: Tensor, grid: Tensor) -> Tensor:
        r"""Estimate density at grid centres via Gaussian KDE.

        .. math::

            \rho(x) = \sum_{p \in \text{points}}
                \exp\!\bigl(-\|x - p\|^2 / (2\sigma^2)\bigr)

        Parameters
        ----------
        points : Tensor
            Point cloud (N, 3).
        grid : Tensor
            Grid centres (G^3, 3).

        Returns
        -------
        Tensor
            Density values (G^3,).
        """
        two_sigma_sq = 2.0 * self.sigma ** 2
        densities = []
        for i in range(0, grid.shape[0], self.chunk_size):
            g_chunk = grid[i : i + self.chunk_size]          # (C, 3)
            # ||x - p||^2 via cdist
            dist_sq = torch.cdist(
                g_chunk.unsqueeze(0), points.unsqueeze(0)
            ).squeeze(0).pow(2)                              # (C, N)
            density = torch.exp(-dist_sq / two_sigma_sq).sum(dim=-1)  # (C,)
            densities.append(density)
        return torch.cat(densities)                          # (G^3,)

    @torch.no_grad()
    def __call__(
        self,
        pred: Tensor,
        gt: Tensor,
    ) -> float:
        """Compute VD between predicted and ground-truth point clouds.

        Parameters
        ----------
        pred : Tensor
            Predicted point cloud (N, 3), coordinates in [-1, 1].
        gt : Tensor
            Ground truth point cloud (M, 3), coordinates in [-1, 1].

        Returns
        -------
        float
            VD score (lower is better).
        """
        grid = self._get_grid(pred.device)

        rho_pred = self._kde_density(pred, grid)
        rho_gt = self._kde_density(gt, grid)

        vd = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for tau in self.thresholds:
            v_pred = (rho_pred > tau).sum().float()
            v_gt = (rho_gt > tau).sum().float()
            vd = vd + torch.abs(v_pred - v_gt) / (v_gt + self.eps)

        return (vd / len(self.thresholds)).item()


# =========================================================================== #
#  Integral Geometry Signature Distance (IGSD)
# =========================================================================== #

def _fibonacci_sphere(M: int, device: torch.device) -> Tensor:
    """Generate M approximately uniform directions on S^2.

    Uses the Fibonacci lattice for low-discrepancy coverage.

    Parameters
    ----------
    M : int
        Number of directions.
    device : torch.device
        Target device.

    Returns
    -------
    Tensor
        Direction unit vectors (M, 3).
    """
    indices = torch.arange(M, dtype=torch.float32, device=device)
    phi = torch.acos(1.0 - 2.0 * (indices + 0.5) / M)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    theta = 2.0 * torch.pi * indices / golden
    return torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi),
    ], dim=-1)


class IntegralGeometrySignatureDistance:
    r"""Integral Geometry Signature Distance (IGSD).

    IGSD measures the consistency of 1-D projection boundaries between
    two point clouds from uniformly sampled viewing angles, forming a
    discrete approximation of the Sliced Wasserstein Distance.  It
    explicitly penalizes topological breaks such as broken thin
    structures.

    For :math:`M` uniformly distributed direction vectors
    :math:`\{u_m\}` on :math:`\mathbb{S}^2`, project both clouds:

    .. math::

        \mathcal{P}^{(m)} = \{ p \cdot u_m \mid p \in X \}

    Sort ascending to obtain quantile signatures :math:`S^{(m)}`, then:

    .. math::

        \mathrm{IGSD}(X_{\text{pred}}, X_{\text{gt}})
        = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{N} \sum_{i=1}^{N}
          \bigl\| S_{\text{pred}}^{(m)}[i]
                - S_{\text{gt}}^{(m)}[i] \bigr\|_2^2

    When :math:`N_{\text{pred}} \neq N_{\text{gt}}`, the quantile
    signatures are linearly interpolated to a common length.

    Lower IGSD = better silhouette and topological consistency.

    Parameters
    ----------
    num_directions : int
        Number of projection directions :math:`M` on the unit sphere.
    """

    def __init__(self, num_directions: int = 128):
        self.M = num_directions
        # Lazily built per device
        self._dir_cache: dict = {}

    def _get_directions(self, device: torch.device) -> Tensor:
        """Return cached direction matrix (M, 3)."""
        key = str(device)
        if key not in self._dir_cache:
            self._dir_cache[key] = _fibonacci_sphere(self.M, device)
        return self._dir_cache[key]

    @torch.no_grad()
    def __call__(
        self,
        pred: Tensor,
        gt: Tensor,
    ) -> float:
        """Compute IGSD between predicted and ground-truth point clouds.

        Parameters
        ----------
        pred : Tensor
            Predicted point cloud (N_pred, 3).
        gt : Tensor
            Ground truth point cloud (N_gt, 3).

        Returns
        -------
        float
            IGSD score (lower is better).
        """
        directions = self._get_directions(pred.device)  # (M, 3)
        N_pred, N_gt = pred.shape[0], gt.shape[0]

        # Project: (N, 3) @ (3, M) -> (N, M), transpose to (M, N)
        proj_pred = (pred @ directions.T).T   # (M, N_pred)
        proj_gt = (gt @ directions.T).T       # (M, N_gt)

        # Sort ascending per direction -> quantile signatures
        sig_pred = proj_pred.sort(dim=-1).values  # (M, N_pred)
        sig_gt = proj_gt.sort(dim=-1).values      # (M, N_gt)

        # If sizes differ, interpolate quantiles to a common length
        if N_pred != N_gt:
            N_common = min(N_pred, N_gt)
            sig_pred = F.interpolate(
                sig_pred.unsqueeze(1),           # (M, 1, N_pred)
                size=N_common,
                mode="linear",
                align_corners=True,
            ).squeeze(1)                         # (M, N_common)
            sig_gt = F.interpolate(
                sig_gt.unsqueeze(1),
                size=N_common,
                mode="linear",
                align_corners=True,
            ).squeeze(1)                         # (M, N_common)

        # Mean squared L2 distance across directions and quantiles
        igsd = ((sig_pred - sig_gt) ** 2).mean()
        return igsd.item()


# =========================================================================== #
#  Functional API (convenience wrappers)
# =========================================================================== #

# Module-level singleton instances for stateless calls
_vd_default = None
_igsd_default = None


def compute_vd(
    pred: Tensor,
    gt: Tensor,
    G: int = 64,
    sigma: float = 0.05,
    thresholds: Sequence[float] = (0.1, 0.3, 0.5, 1.0, 2.0),
    eps: float = 1e-6,
) -> float:
    """Functional API for Valuation Difference.

    See :class:`ValuationDifference` for full documentation.

    Parameters
    ----------
    pred : Tensor
        Predicted point cloud (N, 3), in [-1, 1].
    gt : Tensor
        Ground truth point cloud (M, 3), in [-1, 1].
    G : int
        Grid resolution per axis.
    sigma : float
        KDE bandwidth.
    thresholds : sequence of float
        Density thresholds for volume counting.
    eps : float
        Division stability constant.

    Returns
    -------
    float
        VD score (lower is better).
    """
    metric = ValuationDifference(
        grid_resolution=G,
        sigma=sigma,
        thresholds=thresholds,
        eps=eps,
    )
    return metric(pred, gt)


def compute_igsd(
    pred: Tensor,
    gt: Tensor,
    M: int = 128,
) -> float:
    """Functional API for Integral Geometry Signature Distance.

    See :class:`IntegralGeometrySignatureDistance` for full documentation.

    Parameters
    ----------
    pred : Tensor
        Predicted point cloud (N_pred, 3).
    gt : Tensor
        Ground truth point cloud (N_gt, 3).
    M : int
        Number of projection directions on S^2.

    Returns
    -------
    float
        IGSD score (lower is better).
    """
    metric = IntegralGeometrySignatureDistance(num_directions=M)
    return metric(pred, gt)
