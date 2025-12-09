from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm


def default_slice_transform(resize: int = 299) -> transforms.Compose:
    """Default transform matching Inception preprocessing for 2D slices."""
    return transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_inception_v3(device: str | torch.device = "cpu") -> nn.Module:
    """
    Build an Inception v3 backbone that outputs features before classification.
    """
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def _flatten_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim > 2:
        return torch.flatten(output, 1)
    return output


@torch.no_grad()
def extract_features(
    loader: DataLoader,
    model: nn.Module,
    device: str | torch.device = "cpu",
    aggregate: Literal["mean", "max"] | None = "mean",
    show_progress: bool = False,
) -> np.ndarray:
    """
    Extract features from a volume DataLoader using a slice-wise backbone.

    Expects batches shaped (batch, depth, channels, H, W). Slices are flattened
    into (batch * depth, channels, H, W), passed through the model, then
    aggregated along the depth dimension.
    """
    model.to(device)
    model.eval()

    feats: List[np.ndarray] = []
    iterator: Iterable = tqdm(loader, desc="Extracting features", disable=not show_progress)

    for batch in iterator:
        batch = batch.to(device)
        batch_size, depth, channels, height, width = batch.shape
        batch = batch.view(batch_size * depth, channels, height, width)
        outputs = model(batch)
        outputs = _flatten_output(outputs)
        outputs = outputs.view(batch_size, depth, -1)

        if aggregate == "mean":
            outputs = outputs.mean(dim=1)
        elif aggregate == "max":
            outputs = outputs.max(dim=1).values
        elif aggregate is None:
            outputs = outputs.reshape(batch_size * depth, -1)
        else:
            raise ValueError(f"Unsupported aggregate mode: {aggregate}")

        feats.append(outputs.cpu().numpy())

    return np.vstack(feats)

