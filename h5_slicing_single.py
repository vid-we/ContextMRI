import h5py
import numpy as np
import torch
from pathlib import Path
from mri.fastmri_utils import ifft2c_new

H5_PATH  = "./assets/fastmri/knee/file1001429/file1001429.h5"
OUT_DIR  = Path("./assets/fastmri/knee/file1001429")

def center_crop(img: torch.Tensor, size: int = 320) -> torch.Tensor:
    """Center-Crop eines 2D [H, W] Tensors."""
    h, w = img.shape[0], img.shape[1]
    top  = (h - size) // 2
    left = (w - size) // 2
    return img[top:top+size, left:left+size]

with h5py.File(H5_PATH, "r") as f:
    # Keys
    print("Keys:", list(f.keys()))
    kspace = f["kspace"][:]          # [Slices, H, W] complex64 (single-coil)
    print(f"kspace shape: {kspace.shape}")

n_slices = kspace.shape[0]

sl_dir  = OUT_DIR / "slice"
mps_dir = OUT_DIR / "mps"
sl_dir.mkdir(parents=True, exist_ok=True)
mps_dir.mkdir(parents=True, exist_ok=True)

for sl in range(n_slices):
    kspace_sl = kspace[sl]           # [H, W] complex64

    # Single-Coil: direct IFFT, no sens. map
    kspace_t    = torch.from_numpy(kspace_sl).unsqueeze(0)   # [1, H, W]
    kspace_real = torch.view_as_real(kspace_t)               # [1, H, W, 2]
    img         = torch.view_as_complex(ifft2c_new(kspace_real))  # [1, H, W]
    img         = img.squeeze(0)                             # [H, W]

    # Center-Crop auf 320×320
    img_cropped = center_crop(img, size=320)                 # [320, 320]

    # mps: Single-Coil = Sensitivity = 1
    # Shape [1, 320, 320] for MulticoilMRI
    mps_dummy = torch.ones(1, 320, 320, dtype=torch.complex64)

    np.save(sl_dir  / f"{sl:03d}.npy", img_cropped.numpy())
    np.save(mps_dir / f"{sl:03d}.npy", mps_dummy.numpy())
    print(f"  Slice {sl:03d}: {img_cropped.shape}")

print("Fin")