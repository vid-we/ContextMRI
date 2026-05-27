"""
validate_mri_files.py
=====================
Prüft, ob von h5_slicing.py erzeugte .npy-Files die gleiche Struktur
wie die Referenz-Files haben.

Referenzwerte (aus originalslice_020.npy / originalmps_020.npy):
  slice : (320, 320)  complex64,  99%-Quantil ≈ 1.0  (normiert)
  mps   : (N, 320, 320) complex64, SOS ≈ 1.0 überall

Usage:
  python validate_mri_files.py --slice_dir ./assets/fastmri/knee/file1001429/
                                
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path


# ── Referenz-Constraints (aus originalslice_020.npy / originalmps_020.npy) ──
REF_SLICE_SHAPE  = (320, 320)
REF_SLICE_DTYPE  = np.complex64
REF_MPS_SPATIAL  = (320, 320)          # letzte 2 Dims
REF_MPS_DTYPE    = np.complex64

SLICE_99PCT_TOL  = 0.05   # 99%-Quantil sollte nahe 1.0 sein (±5 %)
SOS_MEAN_TOL     = 0.02   # SOS-Mittelwert sollte nahe 1.0 sein (±2 %)
SOS_MAX_DEV_TOL  = 0.10   # SOS darf max. 10 % von 1.0 abweichen


def check_slice(path: Path) -> list[str]:
    errors = []
    arr = np.load(path)

    if arr.shape != REF_SLICE_SHAPE:
        errors.append(f"shape {arr.shape} != expected {REF_SLICE_SHAPE}")

    if arr.dtype != REF_SLICE_DTYPE:
        errors.append(f"dtype {arr.dtype} != expected {REF_SLICE_DTYPE}")

    if not np.iscomplexobj(arr):
        errors.append("not a complex array")

    pct99 = float(np.percentile(np.abs(arr), 99))
    if abs(pct99 - 1.0) > SLICE_99PCT_TOL:
        errors.append(f"99%-quantile = {pct99:.4f}, expected ≈ 1.0 (+/-{SLICE_99PCT_TOL})")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        errors.append("contains NaN or Inf values")

    return errors


def check_mps(path: Path) -> list[str]:
    errors = []
    arr = np.load(path)

    if arr.ndim != 3:
        errors.append(f"expected 3-D array (coils, H, W), got ndim={arr.ndim}")
        return errors  # weitere Checks wären sinnlos

    if arr.shape[-2:] != REF_MPS_SPATIAL:
        errors.append(f"spatial shape {arr.shape[-2:]} != expected {REF_MPS_SPATIAL}")

    if arr.dtype != REF_MPS_DTYPE:
        errors.append(f"dtype {arr.dtype} != expected {REF_MPS_DTYPE}")

    if not np.iscomplexobj(arr):
        errors.append("not a complex array")

    # SOS (partition-of-unity check)
    sos = np.sqrt(np.sum(np.abs(arr) ** 2, axis=0))   # [H, W]
    sos_mean = float(sos.mean())
    sos_max_dev = float(np.max(np.abs(sos - 1.0)))

    if abs(sos_mean - 1.0) > SOS_MEAN_TOL:
        errors.append(f"SOS mean = {sos_mean:.4f}, expected ≈ 1.0 (+/-{SOS_MEAN_TOL})")

    if sos_max_dev > SOS_MAX_DEV_TOL:
        errors.append(f"SOS max deviation from 1.0 = {sos_max_dev:.4f} (tol {SOS_MAX_DEV_TOL})")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        errors.append("contains NaN or Inf values")

    return errors


def validate_directory(slice_dir: str):
    slice = Path(slice_dir + "slice")
    mps = Path(slice_dir + "mps")
    slice_files = sorted(slice.glob("*.npy")) if slice.exists() else []
    mps_files   = sorted(mps.glob("*.npy"))   if mps.exists()   else []

    if not slice_files:
        print(f"[ERROR] No .npy files found in slice dir: {slice}")
        sys.exit(1)
    if not mps_files:
        print(f"[ERROR] No .npy files found in mps dir: {mps}")
        sys.exit(1)

    # Check that slice and mps files are paired
    slice_names = {f.name for f in slice_files}
    mps_names   = {f.name for f in mps_files}
    only_slice  = slice_names - mps_names
    only_mps    = mps_names   - slice_names
    if only_slice:
        print(f"[WARN]  Slices without matching mps: {sorted(only_slice)}")
    if only_mps:
        print(f"[WARN]  MPS without matching slice:  {sorted(only_mps)}")

    n_ok = n_fail = 0
    for fname in sorted(slice_names & mps_names):
        s_errors = check_slice(slice / fname)
        m_errors = check_mps(mps / fname)

        all_errors = (
            [f"  SLICE: {e}" for e in s_errors] +
            [f"  MPS:   {e}" for e in m_errors]
        )
        if all_errors:
            print(f"[FAIL] {fname}")
            for e in all_errors:
                print(e)
            n_fail += 1
        else:
            n_ok += 1

    print()
    print(f"Results: {n_ok} OK,  {n_fail} FAILED  (out of {n_ok+n_fail} pairs)")
    if n_fail == 0:
        print("All files look correct.")
    else:
        print("Some files have issues.")


def validate_single_pair(slice_path: str, slice_file):
    """Quick check for one specific pair (for ad-hoc testing)."""
    print(f"Checking: {slice_path}")
    s_err = check_slice(slice_path+"/slice/"+slice_file)
    m_err = check_mps(slice_path+"/mps/"+slice_file)

    if not s_err and not m_err:
        print("  OK")
    else:
        for e in s_err:
            print(f"  SLICE: {e}")
        for e in m_err:
            print(f"  MPS:   {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate h5_slicing.py output against reference structure.")
    parser.add_argument("--slice_dir", type=str, default="./assets/fastmri/knee/file1001429/", help="Directory containing slice/*.npy and mps/*.npy files")
    parser.add_argument("--slice_file", type=str, default="020.npy", help="Single slice/mps .npy file")
    args = parser.parse_args()

    if args.slice_file:
        validate_single_pair(args.slice_dir, args.slice_file)
    elif args.slice_dir:
        validate_directory(args.slice_dir)
    else:
        print('No such file/directory', args.slice_dir, args.slice_file)
