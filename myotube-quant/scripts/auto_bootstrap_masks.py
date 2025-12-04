"""
Headless Mask Bootstrapper
---------------------------------------------------------
- Reads images from:   data_root/raw/
- Generates masks to:  data_root/masks/nuclei/<stem>.tif
                       data_root/masks/myotubes/<stem>.tif
- Writes a CSV summary: data_root/bootstrap_summary.csv

Usage:
  python auto_bootstrap_masks.py --data_root /path/to/data_root --min_nuc_area 30 --min_myo_area 500
"""

import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import filters, morphology, exposure, measure
import tifffile as tiff
import csv

def read_rgb(p): return np.array(Image.open(p).convert("RGB"))

def auto_init_masks(rgb, min_nuc_area=30, min_myo_area=500):
    red = exposure.rescale_intensity(rgb[...,0].astype(float), in_range='image', out_range=(0,1))
    blue = exposure.rescale_intensity(rgb[...,2].astype(float), in_range='image', out_range=(0,1))
    r = filters.gaussian(red, 1.2); b = filters.gaussian(blue, 1.0)
    rt, bt = filters.threshold_otsu(r), filters.threshold_otsu(b)
    myo = morphology.remove_small_holes(morphology.remove_small_objects(r>rt, min_size=min_myo_area), area_threshold=256)
    nuc = morphology.remove_small_holes(morphology.remove_small_objects(b>bt, min_size=min_nuc_area), area_threshold=64)
    return nuc.astype(np.uint8), myo.astype(np.uint8)

def label_cc(bin_mask):
    from scipy import ndimage as ndi
    lab, _ = ndi.label(bin_mask.astype(np.uint8))
    return lab

def compute_metrics(nuc, myo):
    nuc_lab = label_cc(nuc)
    myo_lab = label_cc(myo)
    total_nuclei = int(nuc_lab.max())
    myotube_count = int(myo_lab.max())
    # Assign nuclei to myotubes by overlap (any overlap == inside)
    nuclei_props = measure.regionprops(nuc_lab)
    myo_mask = myo_lab > 0
    myHC_positive = 0
    nuclei_myotube_map = {}
    for prop in nuclei_props:
        coords = prop.coords
        inside = np.any(myo_mask[coords[:,0], coords[:,1]])
        if inside: myHC_positive += 1
        mt_ids = myo_lab[coords[:,0], coords[:,1]]
        mt_ids = mt_ids[mt_ids>0]
        if mt_ids.size>0:
            hit = np.bincount(mt_ids).argmax()
            nuclei_myotube_map.setdefault(int(hit), []).append(int(prop.label))
    # fusion index: % nuclei in myotubes with >=2 nuclei
    nuclei_per_mt = [len(v) for v in nuclei_myotube_map.values()]
    fused_nuc = sum(n for n in nuclei_per_mt if n>=2)
    fusion_idx = 100.0 * fused_nuc / total_nuclei if total_nuclei else 0.0
    myHC_pct = 100.0 * myHC_positive / total_nuclei if total_nuclei else 0.0
    avg_n_per_mt = float(np.mean(nuclei_per_mt)) if nuclei_per_mt else 0.0
    return {
        "total_nuclei": total_nuclei,
        "myotube_count": myotube_count,
        "myHC_positive_nuclei": int(myHC_positive),
        "myHC_positive_percentage": round(myHC_pct, 2),
        "nuclei_fused": int(fused_nuc),
        "avg_nuclei_per_myotube": round(avg_n_per_mt, 2),
        "fusion_index": round(fusion_idx, 2),
    }

def save_mask(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), (arr>0).astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--min_nuc_area", type=int, default=30)
    ap.add_argument("--min_myo_area", type=int, default=500)
    args = ap.parse_args()

    root = Path(args.data_root)
    raw = root/"C:/Users/styph/OneDrive/Desktop/Lab/myotube-quant/data/images"
    nuc_dir = root/"masks"/"nuclei"
    myo_dir = root/"masks"/"myotubes"
    nuc_dir.mkdir(parents=True, exist_ok=True); myo_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in raw.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".tif",".tiff"]])
    if not imgs:
        print(f"[ERROR] no images in {raw}")
        sys.exit(1)

    rows = []
    for p in imgs:
        stem = p.stem
        rgb = read_rgb(p)
        nuc, myo = auto_init_masks(rgb, args.min_nuc_area, args.min_myo_area)
        save_mask(nuc, nuc_dir/f"{stem}.tif")
        save_mask(myo, myo_dir/f"{stem}.tif")
        m = compute_metrics(nuc, myo)
        m["image"] = p.name
        rows.append(m)
        print(f"[OK] {p.name} nuclei={m['total_nuclei']} myotubes={m['myotube_count']} fusion_index={m['fusion_index']}%")

    # write summary
    import csv
    out_csv = root/"bootstrap_summary.csv"
    keys = ["image","total_nuclei","myHC_positive_nuclei","myHC_positive_percentage","nuclei_fused",
            "myotube_count","avg_nuclei_per_myotube","fusion_index"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print(f"[DONE] Summary -> {out_csv}")

if __name__ == "__main__":
    main()