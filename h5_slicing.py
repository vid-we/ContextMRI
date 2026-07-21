import h5py
import numpy as np
import torch
import sigpy.mri as mr
from pathlib import Path
from mri.fastmri_utils import ifft2c_new

from datetime import datetime
import time

#H5_PATH = "./assets/fastmri/knee/file1001429/file1001429.h5"
#OUT_DIR = Path("./assets/fastmri/knee/file1001429")

H5_PATH_INIT = "./assets/fastmri/knee/"
OUT_DIR_INIT = Path("./assets/fastmri/knee/")

FILES = ["file1000033", "file1001126", "file1001184",   "file1001429",  "file1001429", "file1001655"]
SLICES = ["010",        "026",         "018",           "020",          "016",         "019"]

def center_crop(img: torch.Tensor, size: int = 320) -> torch.Tensor:
    h, w = img.shape[-2], img.shape[-1]
    top  = (h - size) // 2
    left = (w - size) // 2
    return img[..., top:top+size, left:left+size]

startzeit = datetime.now()
print("Start:", startzeit.strftime("%d.%m.%Y %H:%M:%S"))

for i in range(0, len(FILES)):
    H5_PATH= H5_PATH_INIT+FILES[i]+"/"+FILES[i]+".h5"
    OUT_DIR = OUT_DIR_INIT / FILES[i]

    with h5py.File(H5_PATH, "r") as f:
        kspace = f["kspace"][:]
        print(f"kspace shape: {kspace.shape}")

    n_slices, n_coils = kspace.shape[0], kspace.shape[1]

    for sl in range(n_slices):
        if f"{sl:03d}" == SLICES[i]: #"016":
            kspace_sl = kspace[sl]   # [Coils, H, W]

            # 1. raw k-space to image domain
            kspace_t = torch.from_numpy(kspace_sl)
            kspace_real = torch.view_as_real(kspace_t)

            # Raw full-size coil images
            img_coils_full = torch.view_as_complex(ifft2c_new(kspace_real)) 

            # cropping
            img_coils_cropped = center_crop(img_coils_full, size=320)

            # 320x320 k-space
            from mri.fastmri_utils import fft2c_new
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
            mps_cropped  = center_crop(mps_t, size=320)

            # 5. save
            sl_dir  = OUT_DIR / "slice"
            mps_dir = OUT_DIR / "mps"
            sl_dir.mkdir(parents=True, exist_ok=True)
            mps_dir.mkdir(parents=True, exist_ok=True)

            np.save(sl_dir  / f"{sl:03d}.npy", mvue_cropped.numpy())
            np.save(mps_dir / f"{sl:03d}.npy", mps_cropped.numpy())
            print(f"{FILES[i]} Slice {sl:03d} saved  (scale={scale:.4f})")

endzeit = datetime.now()
print("End :", endzeit.strftime("%d.%m.%Y %H:%M:%S"))