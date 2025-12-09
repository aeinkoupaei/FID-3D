from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

from .data import list_volume_files


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Pick an available device, preferring CUDA if requested."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_feature_file(path: str | Path) -> np.ndarray:
    """Load precomputed features saved as .npy."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected feature array with shape (N, D); got {arr.shape}")
    return arr


def gather_paths_if_needed(paths: Iterable[str | Path] | str | Path) -> List[Path]:
    """
    Accept a directory or iterable of files and return a list of paths.
    """
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            return list_volume_files(p)
        if p.exists():
            return [p]
        raise FileNotFoundError(f"Path not found: {p}")
    return [Path(p) for p in paths]

