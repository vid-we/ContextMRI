# run_this_locally.py  – paste and run from your project root
import numpy as np
from datetime import datetime
import time

FILES = ["file1000033", "file1001126", "file1001184",   "file1001429",  "file1001429", "file1001655"]
SLICES = ["010",        "026",         "018",           "020",          "016",         "019"]

slice="016"
startzeit = datetime.now()
print("Start:", startzeit.strftime("%d.%m.%Y %H:%M:%S"))

for i in range(0, len(FILES)):
    slice = SLICES[i]
    sl  = np.load("./assets/fastmri/knee/"+FILES[i]+"/slice/"+slice+".npy")
    mps = np.load("./assets/fastmri/knee/"+FILES[i]+"/mps/"+slice+".npy")

    #sl  = np.load("./assets/fastmri/knee/file1001429/slice/"+slice+".npy")
    #mps = np.load("./assets/fastmri/knee/file1001429/mps/"+slice+".npy")

    print("# " + FILES[i] +", SLICE "+slice+".npy #")
    print(f"  dtype:            {sl.dtype}")
    print(f"  shape:            {sl.shape}")
    print(f"  is complex:       {np.iscomplexobj(sl)}")
    print(f"  |x| min/max:      {np.abs(sl).min():.6f} / {np.abs(sl).max():.6f}")
    print(f"  99th pct of |x|:  {np.percentile(np.abs(sl), 99):.6f}")

    print()
    print("# " + FILES[i] +", MPS "+slice+".npy #")
    print(f"  dtype:            {mps.dtype}")
    print(f"  shape:            {mps.shape}")
    print(f"  is complex:       {np.iscomplexobj(mps)}")
    sos = np.sqrt(np.sum(np.abs(mps)**2, axis=0))
    print(f"  SOS min/max:      {sos.min():.6f} / {sos.max():.6f}")
    print(f"  SOS mean:         {sos.mean():.6f}")
    print(f"  SOS max dev from 1.0 (all):           {np.max(np.abs(sos - 1.0)):.6f}")
    mask = sos > 0.1
    print(f"  SOS max dev from 1.0 (support >0.1):  {np.max(np.abs(sos[mask] - 1.0)):.6f}")
    print(f"  Fraction in support:                   {mask.mean():.3f}")

endzeit = datetime.now()
print("End :", endzeit.strftime("%d.%m.%Y %H:%M:%S"))