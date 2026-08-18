# Inference & reconstruction support for clip_plain / clip_xml / unimed_clip / schema

This delivery is additive on top of the earlier training-side delivery
(train_mri.py's --conditioning_mode, schema_conditioner.py,
unimed_clip_conditioner.py) - it closes the gap flagged back then:
sampling/reconstruction now works for all four arms, not just clip_plain/clip_xml.

## What's new this turn

- **`pipeline_mri.py`**: `MRIDiffusionPipeline` no longer hardcodes
  CLIP tokenizer/text_encoder calls inside `encode_prompt()`. Two new small
  wrapper classes - `TextEncoderConditioner` (CLIP) and `CallableConditioner`
  (UniMed-CLIP, SchemaConditioner) - give every arm a uniform
  `.encode(prompts, device)` / `.encode_null(batch_size, device)` interface.
  **Backward compatible**: `MRIDiffusionPipeline(text_encoder=..., tokenizer=..., unet=..., scheduler=...)`
  still works exactly as before for clip_plain/clip_xml. New arms pass
  `conditioner=CallableConditioner(...)` instead.
- **`recon_mri.py`**, **`inference.py`**: rebuilt to load whichever
  conditioner `--conditioning_mode` asks for, via the new shared helper.
- **`conditioning_setup.py`** (new): the shared "given `--conditioning_mode`
  and its args, build pipeline-ready kwargs" logic, so recon_mri.py,
  inference.py, and your own sweep script don't each reimplement it.
- **`run_knee_experiments_v3_multiarm.py`**: your uploaded sweep script,
  extended with `conditioning_mode` as a sweep dimension.
- **`utils.py`**: added `user_input_to_metadata_dict()` (schema-mode
  counterpart to the existing `user_input_to_text_string()`), for manual
  single-image inference.

## What I verified vs. what I couldn't

Verified by actually running it (see `tests/`):
- `pipeline.sample()` produces correct-shaped output for all three
  conditioner types (CLIP-style, UniMed-style, schema-style), including the
  classifier-free-guidance null-embedding branch for each.
- The `.to(device)` fix is real, not cosmetic: registering a plain
  `nn.Module` (UniMedCLIPTextEncoder / SchemaConditioner) through
  diffusers' `register_modules()` crashes `pipeline.to(device)` with
  `AttributeError: no attribute 'dtype'` - confirmed against diffusers
  0.39's actual source in my sandbox, not assumed. Worked around by moving
  those modules explicitly in a `to()` override instead of relying on
  diffusers' component-iteration.
- `run_knee_experiments_v3_multiarm.py`'s experiment-list builder: 936
  experiments = 312 x 3 arms, and 312 exactly matches your original v2
  script's count for one arm - confirmed no regression in the sweep logic
  itself. `clip_plain` and `unimed_clip` produce byte-identical prompt
  strings (by design - only the encoder should differ between them).

NOT verified (couldn't be, in my sandbox):
- Actual checkpoint loading for any arm - I have no trained UNet
  checkpoints, no UniMed-CLIP weights, and no huggingface.co access to test
  against. Everything above is validated with fake/randomly-initialized
  encoders standing in for the real ones.
- `recon_mri.py`'s `FIXED_TEST_CASE_METADATA` dict (file1001429/slice 20)
  for schema mode - the field values are copied from your own
  `run_knee_experiments_*.py` SCANS list, not independently re-derived from
  the actual metadata CSV. Sanity-check it if you use `recon_mri.py --conditioning_mode schema`.

## Before running anything

**Every arm needs its own trained checkpoint.** This is the main structural
change from your v2 script: clip_plain, clip_xml, and unimed_clip are three
*separately trained* UNets (from three `train_mri.py --conditioning_mode ...`
runs), not one shared checkpoint with different prompts. Fill in:

```python
# in run_knee_experiments_v3_multiarm.py
CHECKPOINT_PATHS = {
    "clip_plain": "...",
    "unimed_clip": "...",
    "clip_xml": "...",
}
UNIMED_CLIP_CONFIG = {
    "unimed_clip_repo_path": "...",       # your UniMed-CLIP clone
    "unimed_clip_checkpoint": "...",      # downloaded unimed_clip_vit_b16.pt
    ...
}
```
`recon_mri.py --conditioning_mode ... --pretrained_model_name_or_path ...`
and `inference.py` take the equivalent as CLI flags (`--unimed_clip_repo_path`,
`--unimed_clip_checkpoint`, `--metadata_encoder_checkpoint` for schema).

## Run order

```bash
# 1. no checkpoints needed - run first
python tests/test_pipeline_conditioners_smoke.py

# 2. single-image generation, once you have a checkpoint for that arm
python inference.py --conditioning_mode clip_xml \
  --pretrained_model_name_or_path <path> --use_auto True

python inference.py --conditioning_mode schema \
  --pretrained_model_name_or_path <path> \
  --metadata_encoder_checkpoint <path>/metadata_encoder_weights.pth \
  --use_auto False --anatomy knee --slice_number 20 --contrast PDFS

# 3. single fixed-case reconstruction (file1001429/slice 020)
python recon_mri.py --conditioning_mode unimed_clip \
  --pretrained_model_name_or_path <path> \
  --unimed_clip_repo_path <path> --unimed_clip_checkpoint <path>.pt

# 4. the full sweep, once CHECKPOINT_PATHS is filled in
python run_knee_experiments_v3_multiarm.py
```

## Still not covered

`schema` isn't wired into `run_knee_experiments_v3_multiarm.py`'s
`CONDITIONING_MODES` - there's a commented-out
`build_metadata_variants_schema()` sketch at the bottom of that file, but
which fields "no_mr_params" should set to `None` vs. omit is a modeling
choice I didn't want to default silently. `recon_complex_multi.py` (the
batch/eval sibling of `recon_mri.py`) wasn't touched - same pattern would
apply if you want it next.
