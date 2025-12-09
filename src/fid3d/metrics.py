from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance for a feature matrix shaped (N, D)."""
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D (N, D); got shape {features.shape}")
    mu = features.mean(axis=0)
    sigma = cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Frechet distance between two multivariate Gaussians.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Mean vectors have different lengths.")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Covariance matrices must be of the same shape.")

    offset = mu1 - mu2
    eps_eye = np.eye(sigma1.shape[0]) * eps
    covmean = sqrtm((sigma1 + eps_eye) @ (sigma2 + eps_eye))
    if iscomplexobj(covmean):
        covmean = covmean.real

    return float(offset.dot(offset) + trace(sigma1 + sigma2 - 2.0 * covmean))


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray, eps: float = 1e-6) -> float:
    """
    Convenience wrapper: compute FID given real and fake feature arrays.
    """
    mu1, sigma1 = compute_statistics(real_features)
    mu2, sigma2 = compute_statistics(fake_features)
    return frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)

