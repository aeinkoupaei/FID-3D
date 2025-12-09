#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from fid3d.data import VolumeDataset
from fid3d.features import build_inception_v3, default_slice_transform, extract_features
from fid3d.metrics import calculate_fid
from fid3d.utils import get_device, list_volume_files, load_feature_file, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID-3D between two sets of volumes.")
    group_real = parser.add_mutually_exclusive_group(required=True)
    group_real.add_argument("--real", type=str, help="Directory containing real volumes.")
    group_real.add_argument("--real-features", type=str, help="Precomputed real features (.npy).")

    group_fake = parser.add_mutually_exclusive_group(required=True)
    group_fake.add_argument("--fake", type=str, help="Directory containing generated volumes.")
    group_fake.add_argument("--fake-features", type=str, help="Precomputed fake features (.npy).")

    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for DataLoader.")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers for DataLoader.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g., cuda or cpu.")
    parser.add_argument("--resize", type=int, default=299, help="Resize for slice preprocessing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--progress", action="store_true", help="Show progress bars.")
    return parser.parse_args()


def compute_from_dirs(
    real_dir: Path,
    fake_dir: Path,
    batch_size: int,
    num_workers: int,
    device_str: str | None,
    resize: int,
    show_progress: bool,
) -> float:
    device = get_device(prefer_cuda=device_str != "cpu") if device_str else get_device()
    transform = default_slice_transform(resize=resize)
    model = build_inception_v3(device=device)

    real_paths = list_volume_files(real_dir) if real_dir.is_dir() else [real_dir]
    fake_paths = list_volume_files(fake_dir) if fake_dir.is_dir() else [fake_dir]

    real_ds = VolumeDataset(real_paths, slice_transform=transform)
    fake_ds = VolumeDataset(fake_paths, slice_transform=transform)

    loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    real_loader = DataLoader(real_ds, **loader_kwargs)
    fake_loader = DataLoader(fake_ds, **loader_kwargs)

    real_features = extract_features(real_loader, model, device=device, show_progress=show_progress)
    fake_features = extract_features(fake_loader, model, device=device, show_progress=show_progress)
    return calculate_fid(real_features, fake_features)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.real_features and args.fake_features:
        real_features = load_feature_file(args.real_features)
        fake_features = load_feature_file(args.fake_features)
        fid_value = calculate_fid(real_features, fake_features)
    elif args.real and args.fake:
        fid_value = compute_from_dirs(
            Path(args.real),
            Path(args.fake),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_str=args.device,
            resize=args.resize,
            show_progress=args.progress,
        )
    else:
        raise ValueError("Must provide either feature files or volume directories for both real and fake inputs.")

    print(f"FID-3D: {fid_value:.4f}")


if __name__ == "__main__":
    main()

