import json
import numpy as np
from collections import defaultdict

def compute_stats(json_path, version="v1"):
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"\n=== {version.upper()} | Total experiments: {len(data)} ===")
    
    for metric in ['psnr', 'ssim', 'lpips']:
        groups = defaultdict(list)
        for exp in data:
            mask = exp.get("mask_type")
            acc = str(exp.get("acc_factor"))
            cfg = float(exp.get("cfg_scale") or 0.0)
            meta = exp.get("meta_name") or exp.get("Metadata") or "unknown"
            val = exp.get(metric) or exp.get(metric.upper())
            if val is not None and mask:
                key = (mask.lower(), acc, cfg, meta.lower())
                groups[key].append(float(val))
        
        print(f"\n{metric.upper()}:")
        for mask_name, acc_val in [("poisson2d", "8"), ("uniform1d", "4")]:
            print(f"  {mask_name.upper()} {acc_val}:")
            for cfg in [0.0, 1.0, 2.0, 3.0]:
                for meta in ["full", "no_pathology", "anatomy_only", "no_mr_params", "empty"]:
                    key = (mask_name, acc_val, cfg, meta)
                    if key in groups:
                        vals = groups[key]
                        mean = np.mean(vals)
                        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0
                        print(f"    CFG {cfg} {meta}: {mean:.2f} ± {sd:.2f}")

compute_stats("./results/knee_testplan/all_results.json", version='Local Run (v1, N=1 slice)')      # v1
compute_stats("./results/knee_testplan_v2/all_results_v2.json", version='Cloud Run 1 (v2, N=6 slices)')  # v2
compute_stats("./results/knee_testplan_v3/all_results_v3.json", version='Cloud Run 2 (v3, N=6 slices)')  # v3