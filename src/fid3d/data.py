from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = (".npy", ".npz", ".nii", ".nii.gz")


def load_volume(path: str | Path, dtype: np.dtype | str = np.float32) -> np.ndarray:
    """
    Load a 3D volume from .npy/.npz or NIfTI (.nii/.nii.gz).

    Volumes are returned as float arrays with shape (H, W, D).
    """
    volume_path = Path(path)
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume not found: {volume_path}")

    suffix = "".join(volume_path.suffixes) or volume_path.suffix
    if suffix in (".npy", ".npz"):
        arr = np.load(volume_path)
    elif suffix in (".nii", ".nii.gz"):
        arr = np.asanyarray(nib.load(str(volume_path)).get_fdata())
    else:
        raise ValueError(f"Unsupported volume format: {volume_path.suffix}")

    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape} for {volume_path}")
    return arr


def list_volume_files(root: str | Path, extensions: Sequence[str] = SUPPORTED_EXTENSIONS) -> List[Path]:
    """Recursively list volume files under a directory, sorted for reproducibility."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")
    matched = []
    for ext in extensions:
        matched.extend(root_path.rglob(f"*{ext}"))
    return sorted(matched)


class VolumeDataset(Dataset[torch.Tensor]):
    """
    Torch dataset that loads 3D volumes and converts them to stacks of slices.

    Volumes are assumed to be shaped (H, W, D). Each slice is broadcast to
    ``channels`` (default 3) and transformed individually before stacking.
    The returned tensor shape is (D, C, H, W).
    """

    def __init__(
        self,
        volume_paths: Iterable[str | Path],
        slice_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        volume_loader: Callable[[str | Path], np.ndarray] = load_volume,
        channels: int = 3,
    ) -> None:
        self.volume_paths = list(volume_paths)
        if len(self.volume_paths) == 0:
            raise ValueError("No volume files provided to VolumeDataset.")
        self.slice_transform = slice_transform
        self.volume_loader = volume_loader
        self.channels = channels

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        volume = self.volume_loader(self.volume_paths[idx])
        # assume (H, W, D)
        depth = volume.shape[-1]
        slices = []
        for i in range(depth):
            slice_img = torch.from_numpy(volume[..., i]).float().unsqueeze(0)  # (1, H, W)
            if self.channels == 3:
                slice_img = slice_img.repeat(3, 1, 1)
            elif self.channels != 1:
                raise ValueError(f"Unsupported channels={self.channels}; use 1 or 3.")
            if self.slice_transform:
                slice_img = self.slice_transform(slice_img)
            slices.append(slice_img)

        stacked = torch.stack(slices)  # (D, C, H, W)
        return stacked

