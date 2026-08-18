"""
Build metadata_train.csv for the fastMRI knee_singlecoil_train split, in the
same format as ContextMRI's metadata_val.csv.

Two independent sources are combined, per row = one (file, slice):

1. fastMRI raw .h5 files themselves supply: contrast, sequence, TR, TE, TI,
   flip_angle. These live in two places inside every fastMRI HDF5 volume:
     - f.attrs['acquisition']   -> 'CORPD_FBK' / 'CORPDFS_FBK'  (contrast)
     - f['ismrmrd_header'][()]  -> XML text with the full ISMRMRD header
       (sequence name + TR/TE/TI/flip angle under <sequenceParameters>)
   [Zbontar et al. 2018, arXiv:1811.08839, Appendix A;
    Inati et al. 2017, Magn Reson Med 77(1):411-421]

2. fastMRI+ (microsoft/fastmri-plus) Annotations/knee.csv supplies the
   per-slice `pathology` string. This script reproduces the exact join logic
   used to build metadata_val.csv:
       group knee.csv by (file, slice) -> join the `label` column with ", "
       (no de-duplication, no stripping)
   This has been empirically verified against the uploaded metadata_val.csv:
   0/1774 mismatches, including cases with duplicate labels and the raw
   trailing-space artifact in some fastMRI+ label strings (e.g. "Joint
   Effusion "). [Zhao et al. 2022, Sci Data 9:152]

USAGE
-----
    python build_metadata_csv.py \
        --h5_dir /path/to/knee_singlecoil_train \
        --annotation_csv /path/to/fastmri-plus/Annotations/knee.csv \
        --out_csv metadata_train.csv

Before running on the full training set, run --spot_check on ONE file first
(see bottom of this file) and compare the printed values against a val-set
row for a file you recognize in metadata_val.csv. See the "UNVERIFIED /
ASSUMPTIONS" section below for the parts of this script that could NOT be
checked against ground truth and should be spot-checked manually.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import pandas as pd

ISMRMRD_NS = {"m": "http://www.ismrm.org/ISMRMRD"}

# Confirmed via facebookresearch/fastMRI common/evaluate.py --acquisition choices
# and Zbontar et al. 2018 Appendix A ("CORPD"/"CORPDF" = coronal PD with/without
# fat saturation).
ACQUISITION_TO_CONTRAST = {
    "CORPD_FBK": "PD",
    "CORPDFS_FBK": "PDFS",
}


def parse_ismrmrd_header(header_bytes: bytes, anatomy: str = "knee"):
    """Extract sequence name + TR/TE/TI/flip angle from the ISMRMRD XML header.

    VERIFIED against a real file (file1000033): TR=2750, TE=27, TI=100,
    flip_angle=140 extracted here matched metadata_val.csv exactly. Trust
    this part.

    `sequenceName` under measurementInformation, however, is an OPTIONAL
    ISMRMRD field (minOccurs="0") and is simply absent from fastMRI's
    converted headers -- confirmed empirically (file1000033 returns None).
    It is not needed for anatomy == "knee": the `sequence` column is
    literally constant ("TurboSpinEcho") across all 1774 rows of the
    uploaded metadata_val.csv, for both PD and PDFS, and independent
    published descriptions of the fastMRI knee dataset agree the entire
    knee split was acquired with one 2D Turbo Spin Echo protocol across all
    three scanners. So we hardcode it for knee rather than parse a field
    that doesn't exist. This does NOT hold for anatomy == "brain" (AXFLAIR/
    AXT1/AXT1POST/AXT2 are genuinely different sequences there) -- do not
    reuse this hardcode if you extend to brain data.
    """
    root = ET.fromstring(header_bytes)

    def text(path):
        el = root.find(path, ISMRMRD_NS)
        return el.text if el is not None else None

    def all_texts(path):
        return [el.text for el in root.findall(path, ISMRMRD_NS)]

    if anatomy == "knee":
        sequence = "TurboSpinEcho"
        # Cheap sanity check rather than trusting the hardcode blindly: warn
        # (don't crash) if some file's header text doesn't contain the
        # substring anywhere, since that would mean this assumption broke.
        if b"TurboSpinEcho" not in header_bytes:
            print("WARNING: 'TurboSpinEcho' not found anywhere in this "
                  "file's ISMRMRD header text -- hardcoded sequence value "
                  "may be wrong for this volume.")
    else:
        sequence = text(".//m:measurementInformation/m:sequenceName")

    # sequenceParameters children can repeat (e.g. multi-echo protocols).
    # fastMRI knee TSE is expected to report a single value for each; if more
    # than one is present this joins them with ';' rather than silently
    # dropping information, so you'll notice if the assumption is wrong.
    def first_or_joined(vals):
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else ";".join(vals)

    tr = first_or_joined(all_texts(".//m:sequenceParameters/m:TR"))
    te = first_or_joined(all_texts(".//m:sequenceParameters/m:TE"))
    ti = first_or_joined(all_texts(".//m:sequenceParameters/m:TI"))
    flip = first_or_joined(all_texts(".//m:sequenceParameters/m:flipAngle_deg"))

    # metadata_val.csv stores these as bare integers (e.g. "2960", not
    # "2960.0"). Cast defensively; leave as-is if that fails so you can see
    # the raw XML value instead of a crash.
    def to_int_str(v):
        if v is None:
            return None
        try:
            return str(int(round(float(v))))
        except ValueError:
            return v

    return sequence, to_int_str(tr), to_int_str(te), to_int_str(ti), to_int_str(flip)


def load_pathology_lookup(annotation_csv: str) -> pd.Series:
    """Reproduce metadata_val.csv's pathology field exactly.

    Verified against the uploaded metadata_val.csv: 0/1774 rows differ.
    NOTE: fastMRI+ knee.csv contains 13 study_level=='Yes' rows (dataset-wide
    flags such as 'Artifact', recorded at slice 0). None of those 13 rows
    happened to fall inside your val-set files, so whether the original
    script included or excluded them is UNVERIFIED either way -- both
    options reproduce metadata_val.csv perfectly. This script INCLUDES them
    (i.e. does a plain groupby with no filtering), which is the simpler
    hypothesis. If you spot an unexpected 'Artifact' (or similar) pathology
    at slice 0 for some volume, that's this assumption showing up -- filter
    `ann[ann.study_level != 'Yes']` beforehand if you'd rather exclude them.
    """
    ann = pd.read_csv(annotation_csv, keep_default_na=False, na_values=[])
    return ann.groupby(["file", "slice"])["label"].apply(
        lambda s: ", ".join(s.astype(str))
    )


def build(h5_dir: str, annotation_csv: str, out_csv: str, anatomy: str = "knee"):
    pathology_lookup = load_pathology_lookup(annotation_csv)

    rows = []
    h5_files = sorted(Path(h5_dir).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    for h5_path in h5_files:
        filename = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            n_slices = f["kspace"].shape[0]
            acquisition = f.attrs.get("acquisition", "")
            contrast = ACQUISITION_TO_CONTRAST.get(acquisition, acquisition)
            header_bytes = f["ismrmrd_header"][()]
            if isinstance(header_bytes, bytes):
                pass
            else:
                header_bytes = str(header_bytes).encode("utf-8")
            sequence, tr, te, ti, flip = parse_ismrmrd_header(header_bytes, anatomy=anatomy)

        for sl in range(n_slices):
            pathology = pathology_lookup.get((filename, sl), "")
            rows.append(
                {
                    "anatomy": anatomy,
                    "filename": filename,
                    "slice": sl,
                    "contrast": contrast,
                    "pathology": pathology,
                    "sequence": sequence,
                    "TR": tr,
                    "TE": te,
                    "TI": ti,
                    "flip_angle": flip,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows across {len(h5_files)} volumes to {out_csv}")
    return df


def spot_check_one_file(h5_path: str):
    """Print the extracted values for a single volume so you can eyeball them
    against a matching filename in metadata_val.csv before trusting the full
    run over the whole training set."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        n_slices = f["kspace"].shape[0]
        acquisition = f.attrs.get("acquisition", "")
        contrast = ACQUISITION_TO_CONTRAST.get(acquisition, acquisition)
        header_bytes = f["ismrmrd_header"][()]
        sequence, tr, te, ti, flip = parse_ismrmrd_header(header_bytes, anatomy="knee")
    print(f"file: {h5_path.stem}")
    print(f"  n_slices: {n_slices}")
    print(f"  attrs['acquisition']: {acquisition!r} -> contrast: {contrast}")
    print(f"  sequence: {sequence}")
    print(f"  TR: {tr}  TE: {te}  TI: {ti}  flip_angle: {flip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir", type=str, help="Dir of knee_singlecoil_train .h5 files")
    parser.add_argument("--annotation_csv", type=str, help="Path to fastmri-plus Annotations/knee.csv")
    parser.add_argument("--out_csv", type=str, default="metadata_train.csv")
    parser.add_argument("--anatomy", type=str, default="knee")
    parser.add_argument("--spot_check", type=str, default=None,
                         help="Path to a single .h5 file to print+inspect before a full run")
    args = parser.parse_args()

    if args.spot_check:
        spot_check_one_file(args.spot_check)
    else:
        if not args.h5_dir or not args.annotation_csv:
            parser.error("--h5_dir and --annotation_csv are required unless using --spot_check")
        build(args.h5_dir, args.annotation_csv, args.out_csv, args.anatomy)