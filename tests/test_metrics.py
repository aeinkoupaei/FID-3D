import numpy as np

from fid3d.metrics import calculate_fid, compute_statistics, frechet_distance


def test_statistics_shape():
    feats = np.random.randn(10, 4)
    mu, sigma = compute_statistics(feats)
    assert mu.shape == (4,)
    assert sigma.shape == (4, 4)


def test_fid_zero_for_identical_features():
    feats = np.random.randn(8, 3)
    fid = calculate_fid(feats, feats)
    assert np.isclose(fid, 0.0, atol=1e-7)


def test_fid_positive_for_shifted_features():
    feats_real = np.zeros((5, 2))
    feats_fake = np.ones((5, 2))
    fid = calculate_fid(feats_real, feats_fake)
    assert fid > 0


def test_frechet_distance_symmetry():
    rng = np.random.default_rng(0)
    mu1 = rng.standard_normal(3)
    mu2 = rng.standard_normal(3)
    sigma1 = np.eye(3)
    sigma2 = np.eye(3)
    d12 = frechet_distance(mu1, sigma1, mu2, sigma2)
    d21 = frechet_distance(mu2, sigma2, mu1, sigma1)
    assert np.isclose(d12, d21)

