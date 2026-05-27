import h5py
import numpy as np
import torch
import sigpy.mri as mr
from pathlib import Path
from mri.fastmri_utils import ifft2c_new

H5_PATH  = "./assets/fastmri/knee/file1001429/file1001429.h5"
OUT_DIR  = Path("./assets/fastmri/knee/file1001429")

def center_crop(img: torch.Tensor, size: int = 320) -> torch.Tensor:
    # center crop complex image [H, W] to 320x320
    h, w = img.shape[-2], img.shape[-1]
    top  = (h - size) // 2
    left = (w - size) // 2
    return img[..., top:top+size, left:left+size]


with h5py.File(H5_PATH, "r") as f:
    kspace = f["kspace"][:]          # [Slices, Coils, H, W] complex64
    print(f"kspace shape: {kspace.shape}")

n_slices = kspace.shape[0]

for sl in range(n_slices):
    kspace_sl = kspace[sl]           # [Coils, H, W]

    # (ESPiRiT)
    mps = mr.app.EspiritCalib(kspace_sl, calib_width=24, show_pbar=False).run()
    # mps shape: [Coils, H, W] complex64

    # MVUE
    # MulticoilMRI.AT(y) with y = full k space (Mask=1)
    # AT(y) = Σ_i  conj(S_i) · IFFT(y_i)
    #
    # ifft2c_new erwartet [..., H, W, 2] real → gibt [..., H, W, 2] real zurück
    # torch.view_as_real / view_as_complex übernehmen die Konvertierung

    kspace_t = torch.from_numpy(kspace_sl)          # [Coils, H, W] complex64
    mps_t    = torch.from_numpy(mps)                # [Coils, H, W] complex64

    # ── Fix 1: SOS-Normierung ─────────────────────────────────────────────
    # Σ|S_i(x,y)|² soll überall = 1 sein (SENSE-Partition-of-Unity)
    sos = torch.sqrt(torch.sum(torch.abs(mps_t)**2, dim=0, keepdim=True))  # [1, H, W]
    sos = torch.clamp(sos, min=1e-8)   # Division durch Null verhindern
    mps_t = mps_t / sos                # [Coils, H, W], jetzt normiert

    # ── Fix 2: Phasen-Referenz auf Coil 0 setzen ─────────────────────────
    # Alle Coils relativ zur Phase von Coil 0 ausrichten,
    # damit die kohärente Summierung in AT() korrekt funktioniert
    phase_ref = torch.angle(mps_t[0:1, :, :])          # Phase Coil 0: [1, H, W]
    mps_t = mps_t * torch.exp(-1j * phase_ref)          # Phasen-Offset entfernen

    # IFFT to each coil
    # ifft2c_new: real-valued [..., H, W, 2] Input
    kspace_real = torch.view_as_real(kspace_t)      # [Coils, H, W, 2]
    img_coils   = torch.view_as_complex(
        ifft2c_new(kspace_real)                     # [Coils, H, W, 2] → komplex
    )                                               # [Coils, H, W] complex64

    # mult. conjugate sensitifity map - sum over coils
    mvue = torch.sum(torch.conj(mps_t) * img_coils, dim=0)   # [H, W] complex64
    
    # ── NEU: 99%-Quantil-Normierung (exakt wie im Paper) ─────────────────
    scale = float(np.percentile(np.abs(mvue.numpy()), 99))
    mvue  = mvue / scale

    mvue_cropped = center_crop(mvue, size=320)      # [320, 320]
    mps_cropped  = center_crop(mps_t, size=320)     # [Coils, 320, 320]

    # ── Speichern ─────────────────────────────────────────────────────────
    sl_dir  = OUT_DIR / "slice"
    mps_dir = OUT_DIR / "mps"
    sl_dir.mkdir(parents=True, exist_ok=True)
    mps_dir.mkdir(parents=True, exist_ok=True)

    np.save(sl_dir  / f"{sl:03d}.npy", mvue_cropped.numpy())     # z.B. 016.npy
    np.save(mps_dir / f"{sl:03d}.npy", mps_cropped)
    print(f"  Slice {sl:03d} gespeichert")