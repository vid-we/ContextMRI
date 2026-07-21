# run_knee_experiments.py
# Knee MRI reconstructions tests
# - 6 Scans (healthy + diff. pathologies)
# - 4 Mask types
# - 4 diff CFG-Scales
# - 5 Metadata variants

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from pipeline_mri import MRIDiffusionPipeline
from mri.mri import MulticoilMRI
from mri.utils import normalize_np, real_to_nchw_comp, clear, get_mask
from utils import set_seed, calculate_ssim, calculate_lpips
from sigpy.mri import poisson as poisson_mask

from datetime import datetime
import time

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

# MASK CONFIG
#
# Entfernt:  uniform1d x8  (schlechteste Performance im Vortest, ~29 dB)
# Neu:       gaussian2d x8 
#
# uniform1d  x4 (as a baseline)
# gaussian1d x4
# gaussian2d x8, realistische k-Raum-Dichte
# poisson2d  x8, beste Maske aus Vortest, Referenz in paper
MASK_CONFIGS = [
    {"mask_type": "uniform1d",  "acc_factor": 4, "center_fraction": 0.08},
    {"mask_type": "gaussian1d", "acc_factor": 4, "center_fraction": 0.08},
    {"mask_type": "gaussian2d", "acc_factor": 8, "center_fraction": 0.04}, 
    {"mask_type": "poisson2d",  "acc_factor": 8, "center_fraction": 0.04},
]

# CFG-Werte: 0 (leer), 1, 2 3
CFG_SCALES = [0.0, 1.0, 2.0, 3.0]

# ---------------------------------------------------------------------------
# COMMON PARAMETERS
# ---------------------------------------------------------------------------
COMMON = {
    "pretrained_model_name_or_path": "./checkpoints/ContextMRI/MRI_checkpoint",
    "model_config":   "./configs/model_index.json",
    "num_timesteps":  50,
    "eta":            0.8,
    "gamma":          5.0,
    "CG_iter":        5,
    "mri_type":       "fastmri",
    "seed":           42,
    "save_dir":       "./results/knee_testplan_v2",
}

RESULTS_JSON = "./results/knee_testplan_v2/all_results.json"


# METADATA PROMT GENERATION
# Dict {variant_name: prompt_string}

def build_metadata_variants(scan: dict) -> dict:
    s = scan
    anatomy = "Knee"
    sl      = s["slice"]
    contrast = s["contrast"]
    seq     = s["sequence"]
    tr, te, ti, fa = s["TR"], s["TE"], s["TI"], s["flip_angle"]

    # path string (None → no entry)
    if s["pathology"]:
        parts = [p.strip() for p in s["pathology"].split(",")]
        counts = {}
        for p in parts:
            counts[p] = counts.get(p, 0) + 1
        path_str = ", ".join(f"{c} {p}" for p, c in counts.items())
        pathology_clause = f", Pathology: {path_str}"
    else:
        pathology_clause = ""  # healthy

    mr_clause = f", Sequence: {seq}, TR: {tr}, TE: {te}, TI: {ti}, Flip angle: {fa}"

    return {
        # all metadata
        "full":         f"{anatomy}, Slice {sl}, {contrast}{pathology_clause}{mr_clause}",
        # no MR parameters
        "no_mr_params": f"{anatomy}, Slice {sl}, {contrast}{pathology_clause}",
        # no pathology
        "no_pathology": f"{anatomy}, Slice {sl}, {contrast}{mr_clause}",
        # only Anatomie + Kontrast
        "anatomy_only": f"{anatomy}, Slice {sl}, {contrast}",
        # empty prompt equal to cfg_scale=0
        "empty":        "",
    }



# EXPERIMENT LIST
def build_experiment_list() -> list:
    experiments = []
    METADATA_KEEP = {"empty", "full", "no_mr_params", "no_pathology",  "anatomy_only"} # DW, 28.5.2026 for second run with best results from first run
    exp_id = 0
    for scan in SCANS:
        meta_variants = build_metadata_variants(scan)
        for mask_cfg in MASK_CONFIGS:
            for cfg_scale in CFG_SCALES:
                for meta_name, prompt in meta_variants.items():
                    if meta_name not in METADATA_KEEP:
                        continue
                    # empty prompt only with cfg=0
                    if meta_name == "empty" and cfg_scale > 0:
                        continue
                    # cfg=0 only with non emptyp prompt
                    if meta_name != "empty" and cfg_scale == 0:
                        continue
                    exp = {
                        "exp_id":            exp_id,
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


# single EXPERIMENT
def run_single_experiment(exp: dict) -> dict:
    t0 = time.time()
    set_seed(exp["seed"])

    slice_str = f"{exp['slice']:03d}"
    save_dir = (
        Path(exp["save_dir"])
        / exp["filename"]
        / f"slice{slice_str}"
        / exp["mask_type"]
        / f"acc{exp['acc_factor']}"
        / f"cfg{exp['cfg_scale']}"
        / exp["meta_name"]
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, text_encoder, noise_scheduler, unet = _load_models(
        exp["pretrained_model_name_or_path"],
        exp["mri_type"],
        device,
    )
    pipeline = MRIDiffusionPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        config_path=exp["model_config"],
    )
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=exp["num_timesteps"])
    image_size = 320 # for skm-tea 512 would be necessary

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
    print(f"[{exp['exp_id']:03d}] {exp['filename']} sl{exp['slice']:02d} "
          f"{path_tag} | {exp['mask_type']:12s} acc{exp['acc_factor']} "
          f"cfg{exp['cfg_scale']} {exp['meta_name']:15s} → "
          f"PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f} ({runtime:.0f}s)")
    return result



# MODEL CACHE
_model_cache: dict = {}

def _load_models(ckpt_path: str, mri_type: str, device):
    key = (ckpt_path, mri_type)
    if key not in _model_cache:
        tokenizer    = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder").to(device).eval()
        scheduler    = DDPMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
        subfolder    = "fastmri" if mri_type == "fastmri" else "skm-tea"
        unet         = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder=subfolder).to(device).eval()
        _model_cache[key] = (tokenizer, text_encoder, scheduler, unet)
    return _model_cache[key]


# SAVE RESULTS
def save_results(results: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {path}")



# HTML-REPORT
# Gruppiert nach Scan, PSRN
def generate_html_report(results: list, output_path: str):

    def psnr_class(v):
        return "good" if v >= 33 else ("mid" if v >= 30 else "bad")

    sorted_results = sorted(results, key=lambda r: (r["filename"], r["slice"], -r["psnr"]))

    rows = ""
    current_scan = None
    for r in sorted_results:
        scan_key = (r["filename"], r["slice"])
        if scan_key != current_scan:
            current_scan = scan_key
            path_label = r["pathology"] if r["pathology_present"] else "-"
            rows += f"""
        <tr class="scan-header">
          <td colspan="8">{r['filename']} | Slice {r['slice']} | {r['contrast']} | {'Pathology: ' + path_label if r['pathology_present'] else 'healthy'}</td>
        </tr>"""

        rows += f"""
        <tr>
          <td>{r['exp_id']}</td>
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
<title>ContextMRI Knee Testplan</title>
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
<h1>ContextMRI - Knee Testplan v2</h1>
<p>{datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(results)} Experiments | 6 Scans (2 healthy, 4 pathol.) | 4 mask types</p>

<h2>MASK: uniform1d x4 (Baseline) | gaussian1d x4 | gaussian2d x8 | poisson2d x8</h2>
<h2>ALL EXPERIMENTS (sorted by scan,  PSNR)</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>Mask</th><th>Acc</th><th>CFG</th>
      <th>Metadata</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<h2>Summary - average PSNR by mask type</h2>
<table>
  <thead>
    <tr><th>Mask</th><th>Acc</th><th>avg. PSNR (all)</th><th>avg. PSNR (healthy)</th><th>avg. PSNR (pathol.)</th></tr>
  </thead>
  <tbody>
    {''.join(_summary_rows(results))}
  </tbody>
</table>

<p class="generated">Source: {RESULTS_JSON}</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML-Report: {output_path}")


def _summary_rows(results: list) -> list:
    rows = []
    for mc in MASK_CONFIGS:
        mtype = mc["mask_type"]
        acc   = mc["acc_factor"]
        subset  = [r for r in results if r["mask_type"] == mtype and r["acc_factor"] == acc]
        healthy = [r for r in subset if not r["pathology_present"]]
        patho   = [r for r in subset if r["pathology_present"]]
        def avg(lst): return f"{sum(r['psnr'] for r in lst)/len(lst):.2f}" if lst else "—"
        rows.append(
            f"<tr><td>{mtype}</td><td>x{acc}</td>"
            f"<td>{avg(subset)}</td><td>{avg(healthy)}</td><td>{avg(patho)}</td></tr>"
        )
    return rows



# PLOT
# Mittlerer PSNR vs. CFG-Scale über alle 6 Scans

def generate_comparison_plot(results: list, output_path: str):
    fig, axes = plt.subplots(1, len(MASK_CONFIGS), figsize=(5 * len(MASK_CONFIGS), 4))
    if len(MASK_CONFIGS) == 1:
        axes = [axes]

    colors = {"full": "#4fc3f7", "no_mr_params": "#81c784",
              "no_pathology": "#ffb74d", "anatomy_only": "#e57373", "empty": "#aaa"}

    cfg_values = sorted(set(r["cfg_scale"] for r in results))

    for ax, mc in zip(axes, MASK_CONFIGS):
        mtype = mc["mask_type"]
        acc   = mc["acc_factor"]

        for meta in colors:
            points = []
            for cfg in cfg_values:
                subset = [r["psnr"] for r in results
                          if r["mask_type"] == mtype and r["acc_factor"] == acc
                          and r["meta_name"] == meta and r["cfg_scale"] == cfg]
                if subset:
                    points.append((cfg, sum(subset) / len(subset)))
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, "o-", label=meta, color=colors[meta], linewidth=1.5)

        ax.set_title(f"{mtype} x{acc}", fontsize=9)
        ax.set_xlabel("CFG scale")
        ax.set_ylabel("avg. PSNR (dB)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Knee v2: avg. PSNR vs. CFG scale (mean over 6 scans)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved: {output_path}")


# MAIN
def main():
    startzeit = datetime.now()
    print("Start:", startzeit.strftime("%d.%m.%Y %H:%M:%S"))

    experiments = build_experiment_list()
    print(f"Testplan: {len(experiments)} Experiments")
    print(f"Scans:    {len(SCANS)} (2 healthy, 4 pathol.)")
    print(f"Masken:   {len(MASK_CONFIGS)}")
    print(f"CFG:      {CFG_SCALES}")
    print(f"Meta:     5 Variants (full / no_mr_params / no_pathology / anatomy_only / empty)\n")

    results = []

    # Fortsetzen falls unterbrochen
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
            print(f"  ERROR Experiment {exp['exp_id']} ({exp['filename']} sl{exp['slice']}): {e}")
            continue

    report_dir = Path(COMMON["save_dir"])
    generate_html_report(results, str(report_dir / "report.html"))
    generate_comparison_plot(results, str(report_dir / "psnr_comparison.png"))

    if results:
        best = max(results, key=lambda r: r["psnr"])
        print(f"\nBEST result")
        print(f"Scan:      {best['filename']} Slice {best['slice']}")
        print(f"Pathology: {best['pathology'] or '—'}")
        print(f"Mask:      {best['mask_type']} *{best['acc_factor']}")
        print(f"CFG:       {best['cfg_scale']}")
        print(f"Metadata:  {best['meta_name']}")
        print(f"PSNR:      {best['psnr']:.2f} dB")
        print(f"SSIM:      {best['ssim']:.4f}")
        print(f"LPIPS:     {best['lpips']:.4f}")

        # Compare pathology
        print(f"\navg. PSNR: full vs. anatomy_only (pathology status)")
        for has_path in [True, False]:
            label = "pathol." if has_path else "healthy"
            for meta in ["full", "anatomy_only"]:
                subset = [r["psnr"] for r in results
                          if r["pathology_present"] == has_path and r["meta_name"] == meta]
                if subset:
                    print(f"   {label} · {meta:15s}: {sum(subset)/len(subset):.2f} dB (n={len(subset)})")

    endzeit = datetime.now()
    print("End :", endzeit.strftime("%d.%m.%Y %H:%M:%S"))

if __name__ == "__main__":
    main()
