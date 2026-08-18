"""
prepare_knee_singlecoil_subset.py

Turns a folder of downloaded fastMRI `knee_singlecoil_train` .h5 files into
the on-disk layout MRIDataset already expects:
    ./assets/fastmri/knee/<filename>/slice/<slice:03d>.npy   (complex64, 320x320)
plus a metadata CSV with the columns utils.row_to_text_string() reads:
    filename, anatomy, slice, contrast, pathology, sequence, TR, TE, TI, flip_angle

IMPORTANT - read this before running:
1. Singlecoil files have kspace shape (n_slices, H, W) - no coil dimension,
   so unlike h5_slicing.py (which does ESPIRiT + MVUE coil-combination for
   MULTIcoil data), this script does a plain IFFT. No sensitivity maps (mps)
   are produced or needed - MRIDataset.train=True never reads them; they're
   only used later for reconstruction/eval, out of scope for "train on a
   few hundred slices."
2. Metadata: this script FIRST looks for an existing metadata CSV (the
   original ContextMRI repo release / your prior Practical Work most likely
   already has one under ./assets/fastmri/knee/metadata_*.csv - inference.py
   references "./assets/fastmri/knee/metadata_val.csv", so check there
   first). If found, it subsamples rows from it and only extracts the
   matching slices - this is the accurate, recommended path since it reuses
   real contrast/pathology/sequence/TR/TE/TI/flip_angle labels.
   If NOT found, it falls back to building a minimal CSV from the h5 files'
   own headers. That fallback is the part I could not fully verify: the
   top-level `acquisition` attribute (-> contrast) is well documented by the
   fastMRI paper (arXiv:1811.08839), but the ismrmrd_header XML tag names
   for TR/TE/TI/flip_angle were reconstructed from the ISMRMRD schema
   generally, not confirmed against an actual fastMRI knee file's exact XML.
   The script prints a warning and a sample of what it parsed for your first
   file - inspect that before trusting the rest of the run, and compare
   against `python -c "import h5py; f=h5py.File('<path>.h5'); print(f['ismrmrd_header'][()].decode())"`
   on one of your own files if the printed values look wrong (all-NaN, or
   physically implausible TR/TE in ms).

Usage:
    python prepare_knee_singlecoil_subset.py \
        --h5_dir /path/to/knee_singlecoil_train \
        --out_dir ./assets/fastmri/knee \
        --existing_metadata_csv ./assets/fastmri/knee/metadata_val.csv \
        --n_slices 300 --seed 0
"""
import argparse
import glob
import os
import re
import warnings
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd


ISMRMRD_NS = "{http://www.ismrm.org/ISMRMRD}"


def centered_ifft2(kspace: np.ndarray) -> np.ndarray:
    """Standard centered 2D IFFT, applied to the last two axes."""
    x = np.fft.ifftshift(kspace, axes=(-2, -1))
    x = np.fft.ifft2(x, axes=(-2, -1), norm="ortho")
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x


def center_crop(img: np.ndarray, size: int = 320) -> np.ndarray:
    h, w = img.shape[-2], img.shape[-1]
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return img[..., top:top + size, left:left + size]


def acquisition_to_contrast(acquisition: str) -> str:
    """fastMRI's top-level `acquisition` attr for knee is CORPD/CORPDF-family
    strings (arXiv:1811.08839, appendix). ContextMRI's own metadata uses the
    shorter 'PD'/'PDFS' labels (see utils.py / config_mri usage), so we
    normalize known variants and fall back to the raw string (with a
    warning) for anything unrecognized rather than guessing silently."""
    a = acquisition.upper()
    if a in ("CORPD", "CORPD_FBK"):
        return "PD"
    if a in ("CORPDF", "CORPDFS", "CORPDFS_FBK"):
        return "PDFS"
    warnings.warn(f"Unrecognized acquisition string {acquisition!r} - using it as-is for `contrast`. "
                   f"Check against ContextMRI's expected {{'PD','PDFS'}} vocabulary.")
    return acquisition


def parse_sequence_params_best_effort(ismrmrd_header_bytes) -> dict:
    """BEST EFFORT, not verified against a real fastMRI file in the
    environment I wrote this in - see module docstring. Returns a dict with
    possibly-None values rather than raising, so a parsing miss degrades to
    'missing metadata' (which the rest of the pipeline already handles)
    instead of crashing the whole extraction run.

    NOTE: `sequence` is NOT parsed here. Per Table 2 of the ContextMRI paper
    (Chung et al., arXiv:2501.04284), the fastMRI knee set uses a single
    fixed sequence ("Turbospinecho") across the whole dataset - it isn't a
    per-file value worth extracting from XML, so build_minimal_metadata_row()
    hardcodes it directly instead of relying on a guessed tag name here.
    """
    out = {"TR": None, "TE": None, "TI": None, "flip_angle": None}
    try:
        xml_str = ismrmrd_header_bytes.decode("utf-8") if isinstance(ismrmrd_header_bytes, bytes) else ismrmrd_header_bytes
        root = ET.fromstring(xml_str)

        def find_first(tag):
            el = root.find(f".//{ISMRMRD_NS}{tag}")
            if el is None:  # try without namespace, in case the file has none
                el = root.find(f".//{tag}")
            return el.text if el is not None and el.text is not None else None

        tr, te, ti, fa = find_first("TR"), find_first("TE"), find_first("TI"), find_first("flipAngle_deg")
        out["TR"] = float(tr) if tr is not None else None
        out["TE"] = float(te) if te is not None else None
        out["TI"] = float(ti) if ti is not None else None
        out["flip_angle"] = float(fa) if fa is not None else None
    except Exception as e:  # noqa: BLE001 - deliberately broad: this is a best-effort fallback
        warnings.warn(f"ismrmrd_header parsing failed ({e!r}); leaving TR/TE/TI/flip_angle as missing.")
    return out


def build_minimal_metadata_row(h5_path, filename, slice_idx, n_slices_total, print_sample=False):
    with h5py.File(h5_path, "r") as f:
        acquisition = f.attrs.get("acquisition", "UNKNOWN")
        contrast = acquisition_to_contrast(acquisition)
        seq_params = parse_sequence_params_best_effort(f["ismrmrd_header"][()])
        if print_sample:
            print(f"  [sample parse] {filename}: acquisition={acquisition!r} -> contrast={contrast!r}, "
                  f"seq_params={seq_params}")

    return {
        # NOTE: lowercase "knee" - dataset_mri.py builds the .npy path as
        # f"./assets/fastmri/{row['anatomy']}/...", and both h5_slicing.py
        # and inference.py hardcode the lowercase "./assets/fastmri/knee/"
        # folder, so `anatomy` must match that casing here or MRIDataset
        # silently filters out every row (os.path.exists() just returns
        # False, no error - it looks like an empty dataset, not a typo).
        "filename": filename, "anatomy": "knee", "slice": slice_idx,
        "contrast": contrast, "pathology": np.nan,  # fastMRI+ pathology annotations not joined here
        "sequence": "Turbospinecho",  # fixed for the whole fastMRI knee set, see parse_sequence_params_best_effort() docstring
        "TR": seq_params["TR"], "TE": seq_params["TE"],
        "TI": seq_params["TI"], "flip_angle": seq_params["flip_angle"],
    }


def extract_slice_npy(h5_path, slice_idx, out_npy_path, crop_size=320):
    with h5py.File(h5_path, "r") as f:
        kspace_slice = f["kspace"][slice_idx]  # singlecoil: (H, W) complex
    img = centered_ifft2(kspace_slice)
    img = center_crop(img, size=crop_size)
    scale = float(np.percentile(np.abs(img), 99))
    if scale > 0:
        img = img / scale
    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
    np.save(out_npy_path, img.astype(np.complex64))
    return scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir", required=True, help="Folder containing extracted knee_singlecoil_train .h5 files")
    ap.add_argument("--out_dir", default="./assets/fastmri/knee")
    ap.add_argument("--existing_metadata_csv", default=None,
                     help="An existing metadata CSV to subsample from, if you have one "
                          "(check ./assets/fastmri/knee/metadata_*.csv first)")
    ap.add_argument("--n_slices", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop_size", type=int, default=320)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {args.h5_dir}")
    available_filenames = {os.path.splitext(os.path.basename(p))[0] for p in h5_files}
    print(f"Found {len(h5_files)} h5 files under {args.h5_dir}")

    rows = []

    if args.existing_metadata_csv and os.path.exists(args.existing_metadata_csv):
        print(f"Reusing existing metadata CSV: {args.existing_metadata_csv}")
        full_df = pd.read_csv(args.existing_metadata_csv)
        full_df = full_df[full_df["filename"].isin(available_filenames)]
        if full_df.empty:
            raise ValueError(
                f"{args.existing_metadata_csv} has no rows matching filenames found in {args.h5_dir}. "
                f"(Filenames should match between singlecoil/multicoil fastMRI releases for the same "
                f"exam, but double check e.g. full_df['filename'].head() vs your h5 filenames.)"
            )
        n = min(args.n_slices, len(full_df))
        sampled = full_df.sample(n=n, random_state=args.seed).reset_index(drop=True)
        rows = sampled.to_dict("records")
    else:
        print("No existing metadata CSV found/provided - building a MINIMAL one from h5 headers.\n"
              "This fills anatomy/contrast reliably, but pathology is left empty and "
              "sequence/TR/TE/TI/flip_angle are BEST-EFFORT (see this script's docstring).")
        # sample (filename, slice) pairs directly from the h5 files
        candidates = []
        for h5_path in h5_files:
            filename = os.path.splitext(os.path.basename(h5_path))[0]
            with h5py.File(h5_path, "r") as f:
                n_slices_total = f["kspace"].shape[0]
            for s in range(n_slices_total):
                candidates.append((h5_path, filename, s, n_slices_total))
        rng.shuffle(candidates)
        chosen = candidates[: args.n_slices]
        for i, (h5_path, filename, s, n_total) in enumerate(chosen):
            row = build_minimal_metadata_row(h5_path, filename, s, n_total, print_sample=(i == 0))
            rows.append(row)

    print(f"\nExtracting {len(rows)} slices to {args.out_dir} ...")
    filename_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in h5_files}
    kept_rows = []
    for i, row in enumerate(rows):
        filename = row["filename"]
        slice_idx = int(row["slice"])
        h5_path = filename_to_path.get(filename)
        if h5_path is None:
            warnings.warn(f"{filename} not found in {args.h5_dir}, skipping this row.")
            continue
        out_npy = os.path.join(args.out_dir, filename, "slice", f"{slice_idx:03d}.npy")
        try:
            extract_slice_npy(h5_path, slice_idx, out_npy, crop_size=args.crop_size)
            kept_rows.append(row)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Failed to extract {filename} slice {slice_idx}: {e!r}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    out_csv = os.path.join(args.out_dir, f"metadata_train_subset_{len(kept_rows)}.csv")
    pd.DataFrame(kept_rows).to_csv(out_csv, index=False)
    print(f"\nWrote {len(kept_rows)} slices under {args.out_dir}/<filename>/slice/*.npy")
    print(f"Wrote metadata CSV: {out_csv}")
    print(f"\nPass this to train_mri.py as: --mri_metadata_dir_knee {out_csv}")


if __name__ == "__main__":
    main()
