"""PUNet dataloader without pytorch3d dependency.

Replaces pytorch3d.ops.knn_points with pure-torch cdist-based KNN.
Loads .xyz files, adds noise on-the-fly, extracts patches.
"""

import math
import numbers
import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def _knn_points(query: Tensor, reference: Tensor, k: int) -> Tensor:
    """Pure-torch KNN replacement for pytorch3d.ops.knn_points.

    Args:
        query: (B, P, 3) query points.
        reference: (B, N, 3) reference points.
        k: Number of neighbors.

    Returns:
        KNN result points (B, P, K, 3).
    """
    dist = torch.cdist(query, reference)  # (B, P, N)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, P, K)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    nn_pts = torch.gather(
        reference.unsqueeze(1).expand(-1, query.shape[1], -1, -1),
        2, idx_exp,
    )  # (B, P, K, 3)
    return nn_pts


class NormalizeUnitSphere:
    @staticmethod
    def normalize(pcl, center=None, scale=None):
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        data["pcl_clean"], center, scale = self.normalize(data["pcl_clean"])
        data["center"] = center
        data["scale"] = scale
        return data


class AddNoise:
    def __init__(self, noise_std_min, noise_std_max):
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        data["pcl_noisy"] = data["pcl_clean"] + torch.randn_like(data["pcl_clean"]) * noise_std
        data["noise_std"] = noise_std
        return data


class RandomScale:
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data["pcl_clean"] = data["pcl_clean"] * scale
        if "pcl_noisy" in data:
            data["pcl_noisy"] = data["pcl_noisy"] * scale
        return data


class RandomRotate:
    def __init__(self, degrees=180.0, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = torch.tensor(matrix)
        data["pcl_clean"] = torch.matmul(data["pcl_clean"], matrix)
        if "pcl_noisy" in data:
            data["pcl_noisy"] = torch.matmul(data["pcl_noisy"], matrix)
        return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def standard_train_transforms(noise_std_min, noise_std_max, scale_d=0.2, rotate=True):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        RandomScale([1.0 - scale_d, 1.0 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)


class PointCloudDataset(Dataset):
    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, "pointclouds", split, resolution)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        for fn in tqdm(sorted(os.listdir(self.pcl_dir)), desc=f"Loading {resolution}"):
            if not fn.endswith(".xyz"):
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {"pcl_clean": self.pointclouds[idx].clone(), "name": self.pointcloud_names[idx]}
        if self.transform is not None:
            data = self.transform(data)
        return data


def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    """Extract KNN patches from paired point clouds (no pytorch3d needed).

    Args:
        pcl_A: Noisy point cloud (N, 3).
        pcl_B: Clean point cloud (rN, 3).
        patch_size: Points per patch (K).
        num_patches: Number of patches (P).
        ratio: LR/HR ratio.

    Returns:
        pat_A: (P, K, 3), pat_B: (P, rK, 3).
    """
    N = pcl_A.size(0)
    seed_idx = torch.randperm(N)[:num_patches]
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)  # (1, P, 3)

    pat_A = _knn_points(seed_pnts, pcl_A.unsqueeze(0), k=patch_size)
    pat_A = pat_A[0]  # (P, K, 3)

    K_B = int(ratio * patch_size)
    pat_B = _knn_points(seed_pnts, pcl_B.unsqueeze(0), k=K_B)
    pat_B = pat_B[0]  # (P, rK, 3)

    return pat_A, pat_B


class PairedPatchDataset(Dataset):
    def __init__(self, datasets, patch_ratio, on_the_fly=True,
                 patch_size=1000, num_patches=1000, transform=None):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum(len(d) for d in datasets)
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform

    def __len__(self):
        return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        pcl_dset = random.choice(self.datasets)
        pcl_data = pcl_dset[idx % len(pcl_dset)]
        pat_noisy, pat_clean = make_patches_for_pcl_pair(
            pcl_data["pcl_noisy"],
            pcl_data["pcl_clean"],
            patch_size=self.patch_size,
            num_patches=1,
            ratio=self.patch_ratio,
        )
        data = {"pcl_noisy": pat_noisy[0], "pcl_clean": pat_clean[0]}

        if self.transform is not None:
            data = self.transform(data)

        center = data["pcl_clean"].mean(dim=0)
        data["pcl_noisy"] -= center
        data["pcl_clean"] -= center

        scale = torch.max(torch.norm(data["pcl_noisy"], dim=1))
        data["pcl_noisy"] /= scale
        data["pcl_clean"] /= scale

        return {
            "noisy_points": data["pcl_noisy"],
            "clean_points": data["pcl_clean"],
            "center": center,
            "scale": scale,
        }


def get_punet_loaders(
    data_dir: str,
    patch_size: int = 2048,
    batch_size: int = 32,
    noise_min: float = 0.01,
    noise_max: float = 0.02,
    num_workers: int = 4,
    resolutions=None,
    num_patches: int = 200,
):
    """Create train and test dataloaders for PUNet.

    Args:
        data_dir: Path to data/objects/ directory.
        patch_size: Points per patch.
        batch_size: Batch size.
        noise_min: Minimum noise std.
        noise_max: Maximum noise std.
        num_workers: DataLoader workers.
        resolutions: List of resolution folder names.
        num_patches: Patches per model per epoch (controls epoch length).

    Returns:
        Tuple of (train_loader, test_loader).
    """
    if resolutions is None:
        resolutions = ["10000_poisson", "30000_poisson", "50000_poisson"]

    transform = standard_train_transforms(
        noise_std_min=noise_min, noise_std_max=noise_max, rotate=True
    )

    train_ds = PairedPatchDataset(
        datasets=[
            PointCloudDataset(
                root=data_dir, dataset="PUNet", split="train",
                resolution=r, transform=transform,
            )
            for r in resolutions
        ],
        patch_size=patch_size,
        patch_ratio=1.0,
        on_the_fly=True,
        num_patches=num_patches,
    )

    test_ds = PairedPatchDataset(
        datasets=[
            PointCloudDataset(
                root=data_dir, dataset="PUNet", split="test",
                resolution=r, transform=transform,
            )
            for r in resolutions
        ],
        patch_size=patch_size,
        patch_ratio=1.0,
        on_the_fly=True,
        num_patches=num_patches,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, test_loader
