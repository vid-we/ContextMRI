# run_knee_experiments_v3_multiarm.py
#
# Extends your run_knee_experiments_v2_cloud.py with a `conditioning_mode`
# sweep dimension (clip_plain / unimed_clip / clip_xml) so the SAME 6 scans x
# 4 masks x 4 CFG-scales x 5 metadata-variants testplan runs once per arm,
# on each arm's own trained checkpoint.
#
# What changed vs. v2, and why:
#   - CHECKPOINT_PATHS: each arm is a SEPARATELY TRAINED UNet (train_mri.py
#     --conditioning_mode clip_plain / unimed_clip / clip_xml each write
#     their own checkpoint), unlike v2 which assumed one fixed checkpoint
#     for everything. Fill in your three actual paths below.
#   - _load_pipeline() now returns a ready-to-use pipeline via
#     conditioning_setup.build_pipeline_kwargs() instead of always building
#     a CLIPTokenizer/CLIPTextModel pair by hand.
#   - build_metadata_variants_xml(): clip_xml needs the SAME 5 variants as
#     clip_plain, re-tagged as XML (utils.row_to_text_string_xml's tag
#     structure) - reimplemented locally here rather than re-parsing the
#     comma-string, so the XML tags come from the scan dict directly.
#   - exp_id space is now (scan x mask x cfg x meta x conditioning_mode) -
#     3x more experiments than v2 for the same testplan. Consider trimming
#     MASK_CONFIGS/CFG_SCALES/METADATA_KEEP for a first pass across all 3 arms.
#   - schema is NOT included in CONDITIONING_MODES below (no free-text
#     prompt variants map onto it as directly) - see the commented-out
#     build_metadata_variants_schema() sketch near the bottom for the
#     "each variant = a dict with those fields present/absent" version if
#     you have a trained schema checkpoint and want to add it.

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from pipeline_mri import MRIDiffusionPipeline
from mri.mri import MulticoilMRI
from mri.utils import normalize_np, real_to_nchw_comp, clear, get_mask
from utils import set_seed, calculate_ssim, calculate_lpips
from conditioning_setup import build_pipeline_kwargs
from sigpy.mri import poisson as poisson_mask

from datetime import datetime
import time

# ---------------------------------------------------------------------------
# ARMS
# ---------------------------------------------------------------------------
CONDITIONING_MODES = ["clip_plain", "unimed_clip", "clip_xml"]

# FILL IN: each arm's own trained checkpoint directory (from train_mri.py's
# --output_dir, converted via convert_checkpoint.py the same way you already
# do for the baseline).
CHECKPOINT_PATHS = {
    "clip_plain": "./checkpoints/ContextMRI/MRI_checkpoint_clip_plain",
    "unimed_clip": "./checkpoints/ContextMRI/MRI_checkpoint_unimed_clip",
    "clip_xml": "./checkpoints/ContextMRI/MRI_checkpoint_clip_xml",
}

# FILL IN: only used for conditioning_mode == "unimed_clip"
UNIMED_CLIP_CONFIG = {
    "unimed_clip_repo_path": "/path/to/UniMed-CLIP",
    "unimed_clip_checkpoint": "/path/to/unimed_clip_vit_b16.pt",
    "unimed_clip_model_name": "ViT-B-16-quickgelu",
    "unimed_clip_text_encoder_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "unimed_clip_context_length": 77,
}


class _ArgsShim:
    """conditioning_setup.build_pipeline_kwargs() expects an argparse-like
    object; this sweep script has no argparse of its own, so wrap the two
    dicts above into something with the same attribute names."""
    def __init__(self, conditioning_mode):
        self.conditioning_mode = conditioning_mode
        for k, v in UNIMED_CLIP_CONFIG.items():
            setattr(self, k, v)
        self.metadata_encoder_checkpoint = None  # fill in if you add "schema" below


# SCAN-LISTE
# Jeder Eintrag entspricht einem Slice mit echten Metadaten aus der CSV.
# pathology=None  → gesunder Scan (kein Pathologie-Eintrag im Prompt)
SCANS = [
    {
        "filename":    "file1000033",
        "slice":       10,
        "contrast":    "PD",
        "pathology":   None,
        "sequence":    "TurboSpinEcho",
        "TR": 2750, "TE": 27, "TI": 100, "flip_angle": 140,
    },
    {
        "filename":    "file1001126",
        "slice":       26,
        "contrast":    "PD",
        "pathology":   "Cartilage - Partial Thickness loss/defect, Meniscus Tear, Bone- Subchondral edema",
        "sequence":    "TurboSpinEcho",
        "TR": 2750, "TE": 27, "TI": 100, "flip_angle": 140,
    },
    {
        "filename":    "file1001184",
        "slice":       18,
        "contrast":    "PD",
        "pathology":   "Meniscus Tear, Displaced Meniscal Tissue, Ligament - ACL High Grade Sprain",
        "sequence":    "TurboSpinEcho",
        "TR": 2750, "TE": 27, "TI": 100, "flip_angle": 140,
    },
    {
        "filename":    "file1001429",
        "slice":       20,
        "contrast":    "PDFS",
        "pathology":   None,
        "sequence":    "TurboSpinEcho",
        "TR": 3120, "TE": 33, "TI": 100, "flip_angle": 140,
    },
    {
        "filename":    "file1001429",
        "slice":       16,
        "contrast":    "PDFS",
        "pathology":   "Cartilage - Partial Thickness loss/defect, Bone- Subchondral edema",
        "sequence":    "TurboSpinEcho",
        "TR": 3120, "TE": 33, "TI": 100, "flip_angle": 140,
    },
    {
        "filename":    "file1001655",
        "slice":       19,
        "contrast":    "PDFS",
        "pathology":   "Ligament - ACL High Grade Sprain, Bone-Fracture/Contusion/dislocation",
        "sequence":    "TurboSpinEcho",
        "TR": 2870, "TE": 33, "TI": 100, "flip_angle": 140,
    },
]

MASK_CONFIGS = [
    {"mask_type": "uniform1d",  "acc_factor": 4, "center_fraction": 0.08},
    {"mask_type": "gaussian1d", "acc_factor": 4, "center_fraction": 0.08},
    {"mask_type": "gaussian2d", "acc_factor": 8, "center_fraction": 0.04},
    {"mask_type": "poisson2d",  "acc_factor": 8, "center_fraction": 0.04},
]

CFG_SCALES = [0.0, 1.0, 2.0, 3.0]

# ---------------------------------------------------------------------------
# COMMON PARAMETERS
# ---------------------------------------------------------------------------
COMMON = {
    "model_config":   "./configs/model_index.json",
    "num_timesteps":  50,
    "eta":            0.8,
    "gamma":          5.0,
    "CG_iter":        5,
    "mri_type":       "fastmri",
    "seed":           42,
    "save_dir":       "./results/knee_testplan_v3_multiarm",
}

RESULTS_JSON = "./results/knee_testplan_v3_multiarm/all_results.json"


# METADATA PROMPT GENERATION - plain string (clip_plain, unimed_clip)
def build_metadata_variants(scan: dict) -> dict:
    s = scan
    anatomy = "Knee"
    sl      = s["slice"]
    contrast = s["contrast"]
    seq     = s["sequence"]
    tr, te, ti, fa = s["TR"], s["TE"], s["TI"], s["flip_angle"]

    if s["pathology"]:
        parts = [p.strip() for p in s["pathology"].split(",")]
        counts = {}
        for p in parts:
            counts[p] = counts.get(p, 0) + 1
        path_str = ", ".join(f"{c} {p}" for p, c in counts.items())
        pathology_clause = f", Pathology: {path_str}"
    else:
        pathology_clause = ""

    mr_clause = f", Sequence: {seq}, TR: {tr}, TE: {te}, TI: {ti}, Flip angle: {fa}"

    return {
        "full":         f"{anatomy}, Slice {sl}, {contrast}{pathology_clause}{mr_clause}",
        "no_mr_params": f"{anatomy}, Slice {sl}, {contrast}{pathology_clause}",
        "no_pathology": f"{anatomy}, Slice {sl}, {contrast}{mr_clause}",
        "anatomy_only": f"{anatomy}, Slice {sl}, {contrast}",
        "empty":        "",
    }


# METADATA PROMPT GENERATION - XML string (clip_xml), same 5 variants,
# same underlying fields, tag structure matching utils.row_to_text_string_xml.
def build_metadata_variants_xml(scan: dict) -> dict:
    s = scan
    sl, contrast, seq = s["slice"], s["contrast"], s["sequence"]
    tr, te, ti, fa = s["TR"], s["TE"], s["TI"], s["flip_angle"]

    base = f"<anatomy>Knee</anatomy><slice>{sl}</slice><contrast>{contrast}</contrast>"

    if s["pathology"]:
        parts = [p.strip() for p in s["pathology"].split(",")]
        counts = {}
        for p in parts:
            counts[p] = counts.get(p, 0) + 1
        items = "".join(f'<item count="{c}">{p}</item>' for p, c in counts.items())
        pathology_clause = f"<pathology>{items}</pathology>"
    else:
        pathology_clause = ""

    params_clause = (
        f"<params><sequence>{seq}</sequence><TR>{tr}</TR><TE>{te}</TE>"
        f"<TI>{ti}</TI><flip_angle>{fa}</flip_angle></params>"
    )

    return {
        "full":         f"<mri>{base}{pathology_clause}{params_clause}</mri>",
        "no_mr_params": f"<mri>{base}{pathology_clause}</mri>",
        "no_pathology": f"<mri>{base}{params_clause}</mri>",
        "anatomy_only": f"<mri>{base}</mri>",
        "empty":        "",
    }


def metadata_variants_for_mode(scan: dict, conditioning_mode: str) -> dict:
    if conditioning_mode == "clip_xml":
        return build_metadata_variants_xml(scan)
    # clip_plain and unimed_clip: identical prompt strings, only the
    # checkpoint (and therefore encoder) loaded for them differs.
    return build_metadata_variants(scan)


# EXPERIMENT LIST
def build_experiment_list() -> list:
    experiments = []
    METADATA_KEEP = {"empty", "full", "no_mr_params", "no_pathology", "anatomy_only"}
    exp_id = 0
    for conditioning_mode in CONDITIONING_MODES:
        for scan in SCANS:
            meta_variants = metadata_variants_for_mode(scan, conditioning_mode)
            for mask_cfg in MASK_CONFIGS:
                for cfg_scale in CFG_SCALES:
                    for meta_name, prompt in meta_variants.items():
                        if meta_name not in METADATA_KEEP:
                            continue
                        if meta_name == "empty" and cfg_scale > 0:
                            continue
                        if meta_name != "empty" and cfg_scale == 0:
                            continue
                        exp = {
                            "exp_id":            exp_id,
                            "conditioning_mode": conditioning_mode,
                            "pretrained_model_name_or_path": CHECKPOINT_PATHS[conditioning_mode],
                            "filename":          scan["filename"],
                            "slice":             scan["slice"],
                            "pathology":         scan["pathology"],
                            "pathology_present": scan["pathology"] is not None,
                            "contrast":          scan["contrast"],
                            "meta_name":         meta_name,
                            "prompt":            prompt,
                            "cfg_scale":         cfg_scale,
                            **mask_cfg,
                            **COMMON,
                        }
                        experiments.append(exp)
                        exp_id += 1
    return experiments


# MODEL CACHE
_model_cache: dict = {}

def _load_pipeline(exp: dict, device):
    key = (exp["pretrained_model_name_or_path"], exp["mri_type"], exp["conditioning_mode"])
    if key not in _model_cache:
        noise_scheduler = DDPMScheduler.from_pretrained(exp["pretrained_model_name_or_path"], subfolder="scheduler")
        subfolder = "fastmri" if exp["mri_type"] == "fastmri" else "skm-tea"
        unet = UNet2DConditionModel.from_pretrained(exp["pretrained_model_name_or_path"], subfolder=subfolder).to(device).eval()

        args_shim = _ArgsShim(exp["conditioning_mode"])
        pipeline_kwargs = build_pipeline_kwargs(
            args_shim, exp["pretrained_model_name_or_path"],
            cross_attention_dim=unet.config.cross_attention_dim, device=device,
        )
        pipeline = MRIDiffusionPipeline(
            unet=unet, scheduler=noise_scheduler, config_path=exp["model_config"], **pipeline_kwargs,
        )
        pipeline = pipeline.to(device)
        _model_cache[key] = pipeline
    return _model_cache[key]


# single EXPERIMENT
def run_single_experiment(exp: dict) -> dict:
    t0 = time.time()
    set_seed(exp["seed"])

    slice_str = f"{exp['slice']:03d}"
    save_dir = (
        Path(exp["save_dir"])
        / exp["conditioning_mode"]
        / exp["filename"]
        / f"slice{slice_str}"
        / exp["mask_type"]
        / f"acc{exp['acc_factor']}"
        / f"cfg{exp['cfg_scale']}"
        / exp["meta_name"]
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = _load_pipeline(exp, device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=exp["num_timesteps"])
    image_size = 320  # for skm-tea 512 would be necessary

    # load data
    base = f"./assets/fastmri/knee/{exp['filename']}"
    x   = torch.tensor(np.load(f"{base}/slice/{slice_str}.npy")).unsqueeze(0).unsqueeze(0).to(device)
    mps = torch.tensor(np.load(f"{base}/mps/{slice_str}.npy")).unsqueeze(0).to(device)

    B = x.shape[0]
    if exp["mask_type"] == "poisson2d":
        mask = poisson_mask((image_size, image_size), accel=exp["acc_factor"], dtype=np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    else:
        mask = get_mask(
            torch.zeros([B, 1, image_size, image_size]),
            image_size, B,
            type=exp["mask_type"],
            acc_factor=exp["acc_factor"],
            center_fraction=exp["center_fraction"],
        ).to(device)

    prompt  = [exp["prompt"]] if exp["prompt"] else [""]
    A_funcs = MulticoilMRI(image_size, mask, mps)
    y       = A_funcs.A(x)
    ATy     = A_funcs.AT(y)

    recon = pipeline.dds(
        prompt=prompt,
        guidance_scale=exp["cfg_scale"],
        sample_size=image_size,
        num_inference_steps=exp["num_timesteps"],
        eta=exp["eta"],
        A_funcs=A_funcs,
        y=y,
        gamma=exp["gamma"],
        CG_iter=exp["CG_iter"],
    )
    recon = real_to_nchw_comp(recon)

    np.save(str(save_dir / "recon.npy"),    clear(recon))
    plt.imsave(str(save_dir / "recon.png"), np.abs(clear(recon)), cmap="gray")
    plt.imsave(str(save_dir / "label.png"), np.abs(clear(x)),     cmap="gray")
    plt.imsave(str(save_dir / "input.png"), np.abs(clear(ATy)),   cmap="gray")

    x_np     = normalize_np(np.abs(clear(x)))
    recon_np = normalize_np(np.abs(clear(recon)))
    psnr  = peak_signal_noise_ratio(x_np, recon_np)
    ssim  = calculate_ssim(x_np, recon_np)
    lpips = calculate_lpips(x_np, recon_np, device=device)
    runtime = time.time() - t0

    result = {
        "exp_id":            exp["exp_id"],
        "conditioning_mode": exp["conditioning_mode"],
        "filename":          exp["filename"],
        "slice":             exp["slice"],
        "contrast":          exp["contrast"],
        "pathology":         exp["pathology"],
        "pathology_present": exp["pathology_present"],
        "mask_type":         exp["mask_type"],
        "acc_factor":        exp["acc_factor"],
        "cfg_scale":         exp["cfg_scale"],
        "meta_name":         exp["meta_name"],
        "prompt":            exp["prompt"],
        "psnr":              float(psnr),
        "ssim":              float(ssim),
        "lpips":             float(lpips),
        "recon_path":        str(save_dir / "recon.png"),
        "label_path":        str(save_dir / "label.png"),
        "input_path":        str(save_dir / "input.png"),
        "timestamp":         datetime.now().isoformat(),
    }
    path_tag = "pathology" if exp["pathology_present"] else "  healthy  "
    print(f"[{exp['exp_id']:03d}] {exp['conditioning_mode']:11s} {exp['filename']} sl{exp['slice']:02d} "
          f"{path_tag} | {exp['mask_type']:12s} acc{exp['acc_factor']} "
          f"cfg{exp['cfg_scale']} {exp['meta_name']:15s} -> "
          f"PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f} ({runtime:.0f}s)")
    return result


# SAVE RESULTS
def save_results(results: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {path}")


def _summary_rows_by_arm(results: list) -> list:
    rows = []
    for mode in CONDITIONING_MODES:
        for mc in MASK_CONFIGS:
            mtype, acc = mc["mask_type"], mc["acc_factor"]
            subset = [r for r in results if r["conditioning_mode"] == mode
                      and r["mask_type"] == mtype and r["acc_factor"] == acc]
            if not subset:
                continue
            healthy = [r for r in subset if not r["pathology_present"]]
            patho = [r for r in subset if r["pathology_present"]]
            def avg(lst): return f"{sum(r['psnr'] for r in lst)/len(lst):.2f}" if lst else "-"
            rows.append(f"<tr><td>{mode}</td><td>{mtype}</td><td>x{acc}</td>"
                         f"<td>{avg(subset)}</td><td>{avg(healthy)}</td><td>{avg(patho)}</td></tr>")
    return rows


def generate_html_report(results: list, output_path: str):
    def psnr_class(v):
        return "good" if v >= 33 else ("mid" if v >= 30 else "bad")

    sorted_results = sorted(results, key=lambda r: (r["conditioning_mode"], r["filename"], r["slice"], -r["psnr"]))

    rows = ""
    current_scan = None
    for r in sorted_results:
        scan_key = (r["conditioning_mode"], r["filename"], r["slice"])
        if scan_key != current_scan:
            current_scan = scan_key
            path_label = r["pathology"] if r["pathology_present"] else "-"
            rows += f"""
        <tr class="scan-header">
          <td colspan="9">{r['conditioning_mode']} | {r['filename']} | Slice {r['slice']} | {r['contrast']} | {'Pathology: ' + path_label if r['pathology_present'] else 'healthy'}</td>
        </tr>"""
        rows += f"""
        <tr>
          <td>{r['exp_id']}</td>
          <td>{r['conditioning_mode']}</td>
          <td>{r['mask_type']}</td>
          <td>x{r['acc_factor']}</td>
          <td>{r['cfg_scale']}</td>
          <td>{r['meta_name']}</td>
          <td class="{psnr_class(r['psnr'])}">{r['psnr']:.2f}</td>
          <td>{r['ssim']:.4f}</td>
          <td>{r['lpips']:.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>ContextMRI Knee Testplan - Multi-Arm</title>
<style>
  body        {{ font-family: 'Courier New', monospace; background: #0f0f0f; color: #e0e0e0; padding: 2rem; }}
  h1, h2      {{ color: #e0e0e0; font-weight: normal; }}
  h2          {{ font-size: 0.9rem; margin-top: 2rem; color: #aaa; }}
  table       {{ border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.85rem; }}
  th          {{ background: #1e1e1e; padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
  td          {{ padding: 6px 12px; border-bottom: 1px solid #1a1a1a; }}
  tr:hover td {{ background: #1a1a1a; }}
  .scan-header td {{ background: #181818; color: #aaa; padding: 10px 12px 4px; font-size: 0.8rem; border-top: 1px solid #2a2a2a; }}
  .good       {{ font-weight: bold; }}
  .mid        {{ color: #aaa; }}
  .bad        {{ color: #888; }}
  .generated  {{ color: #555; font-size: 0.75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>ContextMRI - Knee Testplan v3 (multi-arm)</h1>
<p>{datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(results)} Experiments | {len(CONDITIONING_MODES)} arms: {', '.join(CONDITIONING_MODES)}</p>

<h2>ALL EXPERIMENTS (sorted by arm, scan, PSNR)</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>Arm</th><th>Mask</th><th>Acc</th><th>CFG</th>
      <th>Metadata</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<h2>Summary - average PSNR by arm x mask type</h2>
<table>
  <thead>
    <tr><th>Arm</th><th>Mask</th><th>Acc</th><th>avg. PSNR (all)</th><th>avg. PSNR (healthy)</th><th>avg. PSNR (pathol.)</th></tr>
  </thead>
  <tbody>
    {''.join(_summary_rows_by_arm(results))}
  </tbody>
</table>

<p class="generated">Source: {RESULTS_JSON}</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML-Report: {output_path}")


# MAIN
def main():
    startzeit = datetime.now()
    print("Start:", startzeit.strftime("%d.%m.%Y %H:%M:%S"))

    experiments = build_experiment_list()
    print(f"Testplan: {len(experiments)} Experiments across {len(CONDITIONING_MODES)} arms: {CONDITIONING_MODES}")
    print(f"Scans:    {len(SCANS)} (2 healthy, 4 pathol.)")
    print(f"Masken:   {len(MASK_CONFIGS)}")
    print(f"CFG:      {CFG_SCALES}\n")

    results = []
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        done_ids = {r["exp_id"] for r in results}
        print(f" {len(done_ids)} Experiments already done, next.\n")
    else:
        done_ids = set()

    for exp in experiments:
        if exp["exp_id"] in done_ids:
            continue
        try:
            result = run_single_experiment(exp)
            results.append(result)
            save_results(results, RESULTS_JSON)
        except Exception as e:
            print(f"  ERROR Experiment {exp['exp_id']} ({exp['conditioning_mode']} {exp['filename']} sl{exp['slice']}): {e}")
            continue

    report_dir = Path(COMMON["save_dir"])
    generate_html_report(results, str(report_dir / "report.html"))

    if results:
        best = max(results, key=lambda r: r["psnr"])
        print(f"\nBEST result overall")
        print(f"Arm:       {best['conditioning_mode']}")
        print(f"Scan:      {best['filename']} Slice {best['slice']}")
        print(f"Mask:      {best['mask_type']} x{best['acc_factor']}")
        print(f"CFG:       {best['cfg_scale']}")
        print(f"Metadata:  {best['meta_name']}")
        print(f"PSNR:      {best['psnr']:.2f} dB")

        print(f"\navg. PSNR by arm (all experiments)")
        for mode in CONDITIONING_MODES:
            subset = [r["psnr"] for r in results if r["conditioning_mode"] == mode]
            if subset:
                print(f"   {mode:12s}: {sum(subset)/len(subset):.2f} dB (n={len(subset)})")

    endzeit = datetime.now()
    print("End :", endzeit.strftime("%d.%m.%Y %H:%M:%S"))


if __name__ == "__main__":
    main()


# --- sketch for adding "schema" as a fourth arm, if you have a trained
# schema checkpoint (train_mri.py --conditioning_mode schema) - not wired
# into CONDITIONING_MODES above because it needs a couple more decisions
# (which fields "no_mr_params" drops as None vs the string arms dropping a
# clause) that are better made deliberately than defaulted here:
#
# def build_metadata_variants_schema(scan: dict) -> dict:
#     s = scan
#     pathologies = [p.strip() for p in s["pathology"].split(",")] if s["pathology"] else []
#     full = {"anatomy": "knee", "slice": float(s["slice"]), "contrast": s["contrast"],
#             "pathologies": pathologies, "sequence": s["sequence"],
#             "TR": s["TR"], "TE": s["TE"], "TI": s["TI"], "flip_angle": s["flip_angle"]}
#     return {
#         "full": full,
#         "no_mr_params": {**full, "sequence": None, "TR": None, "TE": None, "TI": None, "flip_angle": None},
#         "no_pathology": {**full, "pathologies": []},
#         "anatomy_only": {**full, "pathologies": [], "sequence": None, "TR": None, "TE": None, "TI": None, "flip_angle": None},
#         "empty": {},
#     }
