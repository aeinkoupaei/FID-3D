# FID-3D: Frechet Inception Distance for 3D Medical Images

FID-3D provides a clean, modular implementation of the Frechet Inception Distance (FID) adapted to volumetric medical images (e.g., CT/MRI). It loads 3D volumes, extracts features using a configurable backbone, and computes the Frechet distance between real and generated samples. Lower scores indicate closer alignment between generated and real data distributions.

## Why FID for 3D?

FID compares feature statistics (mean and covariance) between two sets of images. Extending this to 3D volumes enables quantitative evaluation of generative models for medical imaging, where per-slice fidelity and volumetric consistency both matter.

## Project Structure

- `src/fid3d/`
  - `data.py` – volume loading utilities and PyTorch dataset for 3D stacks
  - `features.py` – feature extraction helpers and default Inception backbone
  - `metrics.py` – mean/covariance computation and Frechet distance
  - `utils.py` – common helpers (device selection, file discovery, seeds)
  - `__init__.py` – public API exports
- `scripts/compute_fid3d.py` – CLI entry point to compute FID-3D from folders or precomputed features
- `examples/compute_fid_from_dirs.py` – minimal Python example
- `tests/` – small metric tests
- `notebooks/` – place demos or exploratory notebooks
- `requirements.txt` – runtime dependencies

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Usage

### CLI

```bash
python scripts/compute_fid3d.py \
  --real /path/to/real_volumes \
  --fake /path/to/generated_volumes \
  --device cuda
```

Use precomputed features (NumPy `.npy` saved arrays of shape `[N, D]`):

```bash
python scripts/compute_fid3d.py \
  --real-features real_features.npy \
  --fake-features fake_features.npy
```

### Python API

```python
from fid3d.data import VolumeDataset
from fid3d.features import build_inception_v3, extract_features, default_slice_transform
from fid3d.metrics import calculate_fid
from fid3d.utils import list_volume_files
from torch.utils.data import DataLoader

device = "cuda"
model = build_inception_v3(device=device)
transform = default_slice_transform()

real_ds = VolumeDataset(list_volume_files("data/real"), slice_transform=transform)
fake_ds = VolumeDataset(list_volume_files("data/fake"), slice_transform=transform)

real_loader = DataLoader(real_ds, batch_size=1, shuffle=False)
fake_loader = DataLoader(fake_ds, batch_size=1, shuffle=False)

real_feats = extract_features(real_loader, model, device=device)
fake_feats = extract_features(fake_loader, model, device=device)

fid = calculate_fid(real_feats, fake_feats)
print(f"FID-3D: {fid:.4f}")
```

### Swapping Feature Extractors

You can plug in any 2D or 3D PyTorch model. The only requirement is that it returns a feature tensor shaped `[batch, feature_dim]` (or any shape that flattens to that). Replace `build_inception_v3` with your model and keep `extract_features` the same, or author your own aggregation before calling `calculate_fid`.

## Supported Data

- 3D medical volumes stored as `.npy`, `.npz`, `.nii`, or `.nii.gz`
- Volumes are treated as `(H, W, D)` arrays; slices are taken along the last dimension and broadcast to 3 channels before running through a 2D backbone.
