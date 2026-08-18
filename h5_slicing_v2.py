import h5py
import numpy as np
import torch
import sigpy.mri as mr
from pathlib import Path
from mri.fastmri_utils import ifft2c_new, fft2c_new

from datetime import datetime

# --- Input: directory containing the .h5 files directly (no per-file subfolder) ---
IN_DIR = Path("../knee_multicoil_train_batch_0/multicoil_train")

# --- Output: same layout as before, one subfolder per file, containing slice/ and mps/ ---
OUT_DIR_INIT = Path("./assets/fastmri/knee/")

# Set to False to skip writing the mps/*.npy files (e.g. not needed for the training set).
# EspiritCalib still has to run either way, since the MVUE combine needs the coil maps.
SAVE_MPS = False


def center_crop(img: torch.Tensor, size: int = 320) -> torch.Tensor:
    h, w = img.shape[-2], img.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[..., top:top + size, left:left + size]


startzeit = datetime.now()
print("Start:", startzeit.strftime("%d.%m.%Y %H:%M:%S"))

h5_files = sorted(IN_DIR.glob("*.h5"))
print(f"Found {len(h5_files)} h5 files in {IN_DIR}")

for h5_path in h5_files:
    file_stem = h5_path.stem  # e.g. "file1000033"
    OUT_DIR = OUT_DIR_INIT / file_stem

    with h5py.File(h5_path, "r") as f:
        kspace = f["kspace"][:]
        print(f"{file_stem} kspace shape: {kspace.shape}")

    n_slices, n_coils = kspace.shape[0], kspace.shape[1]

    for sl in range(n_slices):
        kspace_sl = kspace[sl]  # [Coils, H, W]

        # 1. raw k-space to image domain
        kspace_t = torch.from_numpy(kspace_sl)
        kspace_real = torch.view_as_real(kspace_t)

        # Raw full-size coil images
        img_coils_full = torch.view_as_complex(ifft2c_new(kspace_real))

        # cropping
        img_coils_cropped = center_crop(img_coils_full, size=320)

        # 320x320 k-space
        img_coils_cropped_real = torch.view_as_real(img_coils_cropped)
        kspace_cropped_t = torch.view_as_complex(fft2c_new(img_coils_cropped_real))
        kspace_cropped_np = kspace_cropped_t.numpy()

        # 2. espirit
        mps = mr.app.EspiritCalib(
            kspace_cropped_np,
            calib_width=24,
            crop=0.95,
            show_pbar=False,
        ).run()

        # 3. MVUE
        mps_t = torch.from_numpy(mps)
        mvue_cropped = torch.sum(torch.conj(mps_t) * img_coils_cropped, dim=0)

        # 4. Scale (99th percentile)
        scale = float(np.percentile(np.abs(mvue_cropped.numpy()), 99))
        mvue_cropped = mvue_cropped / scale

        # 5. save
        sl_dir = OUT_DIR / "slice"
        sl_dir.mkdir(parents=True, exist_ok=True)
        np.save(sl_dir / f"{sl:03d}.npy", mvue_cropped.numpy())

        if SAVE_MPS:
            mps_cropped = center_crop(mps_t, size=320)
            mps_dir = OUT_DIR / "mps"
            mps_dir.mkdir(parents=True, exist_ok=True)
            np.save(mps_dir / f"{sl:03d}.npy", mps_cropped.numpy())

        print(f"{file_stem} Slice {sl:03d} saved  (scale={scale:.4f})")

endzeit = datetime.now()
print("End :", endzeit.strftime("%d.%m.%Y %H:%M:%S"))
