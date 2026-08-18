# Three-arm comparison: clip_plain vs unimed_clip vs clip_xml

## What changed vs. the previous delivery

- `train_mri.py` now has a 4-way `--conditioning_mode`: `clip_plain`, `clip_xml`,
  `unimed_clip` (new), `schema`. All the plumbing (optimizer groups, checkpoint
  save/load, gradient clipping, `accelerator.prepare`) was reworked from a
  binary switch into a proper dispatch - see the `uses_clip` / `uses_unimed` /
  `uses_schema` / `trainable_conditioner` variables near the top of `main()`.
- `unimed_clip_conditioner.py` (new): wraps UniMed-CLIP's BiomedBERT text tower
  so it's a drop-in replacement for OpenAI CLIP's `encoder_hidden_states`.
- `prepare_knee_singlecoil_subset.py` (new): builds a few-hundred-slice subset
  from `knee_singlecoil_train` .h5 files, in the layout `MRIDataset` expects.
- Fixed a real bug while testing: `train_mri.py` used to hard-require both
  `--mri_metadata_dir_knee` and `--mri_metadata_dir_brain`; relaxed to only
  require knee, since `MRIDataset` already treats a missing brain CSV as
  "no brain data" rather than an error. Needed for a knee-only run.

## Step 0 - install UniMed-CLIP (only needed for the unimed_clip arm)

```bash
git clone https://github.com/mbzuai-oryx/UniMed-CLIP.git
cd UniMed-CLIP && pip install -r requirements.txt && cd ..
```
Then download `unimed_clip_vit_b16` (the base-text-encoder checkpoint) from
the SharePoint link in that repo's README "Pre-trained Models" table - this
is a manual download, there's no wget one-liner for it. Note the local path
to the downloaded `.pt` file.

**Verify the pairing before training**: `ViT-B-16-quickgelu` +
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` is what keeps the text
encoder at 768-dim (matching your UNet, no projection layer). Don't swap in
a `ViT-L-14` checkpoint without adding a projection - see
`unimed_clip_conditioner.py`'s docstring.

## Step 1 - build the few-hundred-slice subset

```bash
python prepare_knee_singlecoil_subset.py \
  --h5_dir /path/to/knee_singlecoil_train \
  --out_dir ./assets/fastmri/knee \
  --existing_metadata_csv ./assets/fastmri/knee/metadata_val.csv \
  --n_slices 300 --seed 0
```
Drop `--existing_metadata_csv` if you don't have one - the script falls back
to building a minimal CSV from the h5 headers itself (read the warnings it
prints; TR/TE/TI/flip_angle extraction there is best-effort, see the script's
docstring). This writes `./assets/fastmri/knee/metadata_train_subset_300.csv`
- use that path below.

Sanity-check before spending GPU time - this needs only the file from step 1,
no encoders, no torch training loop:
```bash
python -c "
from dataset_mri import MRIDataset
ds = MRIDataset(metadata_file_knee='./assets/fastmri/knee/metadata_train_subset_300.csv',
                 metadata_file_brain=None, train=True, conditioning_mode='clip_plain')
print(len(ds), ds[0]['prompt'])
"
```

## Step 2 - the three runs

Same data, same UNet config, same seed for all three - only `--conditioning_mode`
(and the two `unimed_clip_*` args, ignored by the other two modes) changes.

```bash
SUBSET_CSV=./assets/fastmri/knee/metadata_train_subset_300.csv
PRETRAINED=<path-or-hub-id-with-tokenizer/text_encoder/scheduler-subfolders>

# 1) baseline - original ContextMRI as-is
accelerate launch train_mri.py \
  --pretrained_model_name_or_path $PRETRAINED \
  --mri_metadata_dir_knee $SUBSET_CSV \
  --conditioning_mode clip_plain \
  --output_dir ./runs/clip_plain \
  --seed 0 --train_batch_size 4 --num_train_epochs 20 --checkpointing_steps 200

# 2) encoder swap - UniMed-CLIP instead of CLIP, SAME string as baseline
accelerate launch train_mri.py \
  --pretrained_model_name_or_path $PRETRAINED \
  --mri_metadata_dir_knee $SUBSET_CSV \
  --conditioning_mode unimed_clip \
  --unimed_clip_repo_path /path/to/UniMed-CLIP \
  --unimed_clip_checkpoint /path/to/unimed_clip_vit_b16.pt \
  --output_dir ./runs/unimed_clip \
  --seed 0 --train_batch_size 4 --num_train_epochs 20 --checkpointing_steps 200

# 3) structured string - XML through the SAME CLIP as baseline
accelerate launch train_mri.py \
  --pretrained_model_name_or_path $PRETRAINED \
  --mri_metadata_dir_knee $SUBSET_CSV \
  --conditioning_mode clip_xml \
  --output_dir ./runs/clip_xml \
  --seed 0 --train_batch_size 4 --num_train_epochs 20 --checkpointing_steps 200
```

Notes on the flags I picked:
- `--num_train_epochs 20` for ~300 slices / batch 4 is a guess to get enough
  gradient steps out of a small subset (~1500 steps) - adjust based on how
  fast loss actually moves; I have no loss curve to calibrate against.
- Drop `accelerate launch` for a single-process/CPU sanity run and just call
  `python train_mri.py ...` directly - `Accelerator()` still works without
  `accelerate config` having been run.
- Do a `--max_train_steps 5` dry run of each command first (see the previous
  delivery's `README_ARM_TESTING.md`) before committing to the full epoch count.

## What to compare afterward

Each `./runs/<arm>/` gets its own `unet_weights.pth` / `unet_ema_0.999_weights.pth`
checkpoints - `unimed_clip` doesn't produce a separate encoder checkpoint since
that encoder is frozen (same as `clip_plain`/`clip_xml`). `pipeline_mri.py` /
`inference.py` aren't updated for any of these three modes yet for
sampling/reconstruction - that's the next piece if you want to actually
generate or reconstruct images with these checkpoints rather than just
compare training loss curves.
