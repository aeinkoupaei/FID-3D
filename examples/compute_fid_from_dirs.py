from pathlib import Path

from torch.utils.data import DataLoader

from fid3d.data import VolumeDataset
from fid3d.features import build_inception_v3, default_slice_transform, extract_features
from fid3d.metrics import calculate_fid
from fid3d.utils import get_device, list_volume_files, set_seed


def main() -> None:
    set_seed(7)
    device = get_device()

    real_paths = list_volume_files(Path("data/real"))
    fake_paths = list_volume_files(Path("data/fake"))

    transform = default_slice_transform()
    real_ds = VolumeDataset(real_paths, slice_transform=transform)
    fake_ds = VolumeDataset(fake_paths, slice_transform=transform)

    loader_kwargs = dict(batch_size=1, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    real_loader = DataLoader(real_ds, **loader_kwargs)
    fake_loader = DataLoader(fake_ds, **loader_kwargs)

    model = build_inception_v3(device=device)
    real_feats = extract_features(real_loader, model, device=device)
    fake_feats = extract_features(fake_loader, model, device=device)

    fid = calculate_fid(real_feats, fake_feats)
    print(f"FID-3D: {fid:.4f}")


if __name__ == "__main__":
    main()

