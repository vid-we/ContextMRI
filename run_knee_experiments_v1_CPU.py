# run_knee_experiments.py
# tests für knee MRI Reconstr.
# - different masktypes, acceleration factors, CFG values, metadata

import subprocess
import json
import os
import itertools
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio
from mri.utils import normalize_np, real_to_nchw_comp, clear, get_mask

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from pipeline_mri import MRIDiffusionPipeline
from mri.mri import MulticoilMRI
# from mri.utils import get_mask
from utils import set_seed, calculate_ssim, calculate_lpips
from sigpy.mri import poisson as poisson_mask

#from transformers import CLIPTokenizer, CLIPTextModel
#from diffusers import DDPMScheduler, UNet2DConditionModel


# TEST DEFINITION
# masktypes, acceleration factors
MASK_CONFIGS = [
    {"mask_type": "uniform1d",  "acc_factor": 4,  "center_fraction": 0.08},
    {"mask_type": "uniform1d",  "acc_factor": 8,  "center_fraction": 0.04},
    {"mask_type": "gaussian1d", "acc_factor": 4,  "center_fraction": 0.08},
    {"mask_type": "poisson2d",  "acc_factor": 8,  "center_fraction": 0.04},
]

# CFG values
CFG_SCALES = [0.0, 1.0, 2.0, 3.0]

# metadata
METADATA_VARIANTS = {
    "full":         "Knee, Slice 20, PDFS, Pathology: Meniscus Tear, Sequence: Turbospinecho, TR: 3150, TE: 33, TI: 100, Flip angle: 150",
    "no_mr_params": "Knee, Slice 20, PDFS, Pathology: Meniscus Tear",
    "no_pathology": "Knee, Slice 20, PDFS, Sequence: Turbospinecho, TR: 3150, TE: 33, TI: 100, Flip angle: 150",
    "anatomy_only": "Knee, Slice 20, PDFS",
    "empty":        "",   # equal cfg_scale=0
}

# default parameters
COMMON = {
    "pretrained_model_name_or_path": "./checkpoints/ContextMRI/MRI_checkpoint",
    "model_config":    "./configs/model_index.json",
    "num_timesteps":   50,
    "eta":             0.8,
    "gamma":           5.0,
    "CG_iter":         5,
    "mri_type":        "fastmri",
    "seed":            42,
    "save_dir":        "./results/knee_testplan",
} 

RESULTS_JSON = "./results/knee_testplan/all_results.json"


# helpers
def build_experiment_list():
    # returns Dict-list -> full run
    experiments = []
    exp_id = 0
    for mask_cfg in MASK_CONFIGS:
        for cfg_scale in CFG_SCALES:
            for meta_name, prompt in METADATA_VARIANTS.items():
                # emtpy promp -> jump
                if meta_name == "empty" and cfg_scale > 0:
                    continue
                # non empty prompt but cfg=0 -> jump
                if meta_name != "empty" and cfg_scale == 0:
                    continue
                exp = {
                    "exp_id":    exp_id,
                    "meta_name": meta_name,
                    "prompt":    prompt,
                    "cfg_scale": cfg_scale,
                    **mask_cfg,
                    **COMMON,
                }
                experiments.append(exp)
                exp_id += 1
    return experiments


def run_single_experiment(exp: dict) -> dict:
    # returns dict with metrics
    
    set_seed(exp["seed"])
    # output path
    save_dir = (
        Path(exp["save_dir"])
        / exp["mask_type"]
        / f"acc{exp['acc_factor']}"
        / f"cfg{exp['cfg_scale']}"
        / exp["meta_name"]
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model (caching after 1st run via @lru_cache)
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
    image_size = 512 if exp["mri_type"] == "skm-tea" else 320

    # load data
    x   = torch.tensor(np.load("./assets/fastmri/knee/file1001429/slice/020.npy")).unsqueeze(0).unsqueeze(0)
    mps = torch.tensor(np.load("./assets/fastmri/knee/file1001429/mps/020.npy")).unsqueeze(0)
    x   = x.to(device)
    mps = mps.to(device)
    # Mask
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
    # Prompt
    prompt = [exp["prompt"]] if exp["prompt"] else [""]
    A_funcs = MulticoilMRI(image_size, mask, mps)
    y   = A_funcs.A(x)
    ATy = A_funcs.AT(y)
    # Rekonstr.
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
    np.save(str(save_dir / "recon.npy"),        clear(recon))
    plt.imsave(str(save_dir / "recon.png"),     np.abs(clear(recon)), cmap="gray")
    plt.imsave(str(save_dir / "label.png"),     np.abs(clear(x)),     cmap="gray")
    plt.imsave(str(save_dir / "input.png"),     np.abs(clear(ATy)),   cmap="gray")
    # metrics
    x_np    = normalize_np(np.abs(clear(x)))
    recon_np = normalize_np(np.abs(clear(recon)))
    psnr  = peak_signal_noise_ratio(x_np, recon_np)
    ssim  = calculate_ssim(x_np, recon_np)
    lpips = calculate_lpips(x_np, recon_np, device=device)
    result = {
        "exp_id":      exp["exp_id"],
        "mask_type":   exp["mask_type"],
        "acc_factor":  exp["acc_factor"],
        "cfg_scale":   exp["cfg_scale"],
        "meta_name":   exp["meta_name"],
        "prompt":      exp["prompt"],
        "psnr":        float(psnr),
        "ssim":        float(ssim),
        "lpips":       float(lpips),
        "recon_path":  str(save_dir / "recon.png"),
        "label_path":  str(save_dir / "label.png"),
        "input_path":  str(save_dir / "input.png"),
        "timestamp":   datetime.now().isoformat(),
    }
    print(f"[{exp['exp_id']:03d}] {exp['mask_type']} acc{exp['acc_factor']} "
          f"cfg{exp['cfg_scale']} {exp['meta_name']:15s} → "
          f"PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f}")
    return result

# model cache
_model_cache = {}
def _load_models(ckpt_path, mri_type, device):

    key = (ckpt_path, mri_type)
    if key not in _model_cache:
        tokenizer     = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
        text_encoder  = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder").to(device).eval()
        scheduler     = DDPMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
        subfolder     = "fastmri" if mri_type == "fastmri" else "skm-tea"
        unet          = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder=subfolder).to(device).eval()
        _model_cache[key] = (tokenizer, text_encoder, scheduler, unet)
    return _model_cache[key]


def save_results(results: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nErgebnisse gespeichert: {path}")



# REPORT
def generate_html_report(results: list, output_path: str):

    def psnr_class(v):
        return "good" if v >= 33 else ("mid" if v >= 30 else "bad")

    sorted_results = sorted(results, key=lambda r: (-r["psnr"]))

    rows = ""
    current_scan = None
    for r in sorted_results:
        scan_key = ('file1001429 | Slice 20')
        if scan_key != current_scan:
            current_scan = scan_key
            rows += f"""
            <tr class="scan-header">
            <td colspan="8">{'file1001429 | Slice 20'}</td>
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
<h1>ContextMRI - Knee Testplan v1</h1>
<p>{datetime.now().strftime('%Y-%m-%d %H:%M')} | {len(results)} Experiments | 1 Scan | 4 mask types</p>
uniform1d
<h2>MASK: uniform1d x4 (Baseline) | uniform1d x8 | gaussian1d x4 | poisson2d x8</h2>
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
    <tr><th>Mask</th><th>Acc</th><th>avg. PSNR (all)</th></tr>
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
        def avg(lst): return f"{sum(r['psnr'] for r in lst)/len(lst):.2f}" if lst else "—"
        rows.append(
            f"<tr><td>{mtype}</td><td>x{acc}</td>"
            f"<td>{avg(subset)}</td></tr>"
        )
    return rows



def generate_comparison_plot(results: list, output_path: str):
    # PSNR vs. CFG scale based on mask type
    fig, axes = plt.subplots(1, len(MASK_CONFIGS), figsize=(5 * len(MASK_CONFIGS), 4))
    if len(MASK_CONFIGS) == 1:
        axes = [axes]

    colors = {"full": "#4fc3f7", "no_mr_params": "#81c784",
              "no_pathology": "#ffb74d", "anatomy_only": "#e57373", "empty": "#aaa"}

    for ax, mask_cfg in zip(axes, MASK_CONFIGS):
        mtype = mask_cfg["mask_type"]
        acc   = mask_cfg["acc_factor"]

        for meta_name in METADATA_VARIANTS:
            subset = [
                r for r in results
                if r["mask_type"] == mtype
                and r["acc_factor"] == acc
                and r["meta_name"] == meta_name
            ]
            if not subset:
                continue
            xs = [r["cfg_scale"] for r in subset]
            ys = [r["psnr"]      for r in subset]
            ax.plot(xs, ys, "o-", label=meta_name,
                    color=colors.get(meta_name, "#fff"), linewidth=1.5)

        ax.set_title(f"{mtype} * {acc}", fontsize=9)
        ax.set_xlabel("CFG scale")
        ax.set_ylabel("PSNR (dB)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Knee: PSNR vs. CFG scale", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved under: {output_path}")





# MAIN
def main():
    
    experiments = build_experiment_list()
    print(f"Testplan: {len(experiments)} experiments\n")
    for e in experiments:
        print(f"  [{e['exp_id']:03d}] {e['mask_type']:12s} acc{e['acc_factor']} "
              f"cfg{e['cfg_scale']} meta={e['meta_name']}")
    results = []

    # finished results (if continued)
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        done_ids = {r["exp_id"] for r in results}
        print(f"  {len(done_ids)} experiments already done, moving on.")
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
            print(f"  ERROR with experiment {exp['exp_id']}: {e}")
            continue
          
    # Report
    report_dir = Path(COMMON["save_dir"])
    generate_html_report(results, str(report_dir / "report.html"))
    generate_comparison_plot(results, str(report_dir / "psnr_comparison.png"))

    # Best result
    if results:
        best = max(results, key=lambda r: r["psnr"])
        print(f"\n=== BEST RESULT ===")
        print(f"   Maske:     {best['mask_type']} * {best['acc_factor']}")
        print(f"   CFG:       {best['cfg_scale']}")
        print(f"   Metadaten: {best['meta_name']}")
        print(f"   PSNR:      {best['psnr']:.2f} dB")
        print(f"   SSIM:      {best['ssim']:.4f}")
        print(f"   LPIPS:     {best['lpips']:.4f}")


if __name__ == "__main__":
    main()
