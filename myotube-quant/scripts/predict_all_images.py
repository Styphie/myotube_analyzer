"""
predict_all_images.py
---------------------
Load a trained U-Net model and run it on *all* images in data/images/,
compute biological metrics from the predicted masks, and save them to
metrics_all.csv. This lets you compare directly to manual Excel
metrics with metrics_eval.py.

Usage (example):

  python scripts/predict_all_images.py \
    --data_root "C:/Users/styph/OneDrive/Desktop/Lab/myotube-quant/data" \
    --model_path "runs/20251201_171728/model_best.pt" \
    --image_size 512 \
    --out_csv "runs/20251201_171728/metrics_all.csv"
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch

# Import classes and helpers from the training script
from train_myotube_nuclei_unet import (
    UNet,
    compute_bio_metrics,
    set_seed,
    read_rgb,
    postprocess_masks
)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to data/ folder")
    ap.add_argument("--model_path", required=True, help="Path to model_best.pt")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--out_csv", required=True, help="Where to write metrics CSV")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    img_dir = data_root / "images"

    images = sorted(
        [p for p in img_dir.glob("*")
         if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]]
    )
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # Load model
    ckpt = torch.load(args.model_path, map_location=device)
    model = UNet(in_ch=2, out_ch=2, base=32)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    rows = []
    out_pred_dir = Path(args.out_csv).parent / "pred_all_images"
    ensure_dir(out_pred_dir)

    for p in images:
        stem = p.stem
        # --- build input (same as in MyoDataset) ---
        rgb = read_rgb(p).astype(np.float32) / 255.0
        red = rgb[..., 0]
        blue = rgb[..., 2]
        H = W = args.image_size
        red_r = np.array(Image.fromarray(red).resize((W, H), Image.BILINEAR))
        blue_r = np.array(Image.fromarray(blue).resize((W, H), Image.BILINEAR))
        x = np.stack([red_r, blue_r], axis=0)  # [2, H, W]
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # [1, 2, H, W]

        with torch.no_grad():
            logits = model(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # [2, H, W]

        pr = (probs > 0.5).astype(np.uint8)
        nuc_pred = pr[0]
        myo_pred = pr[1]

        # ---------------------------------------------------
        # Clean up predictions
        # ---------------------------------------------------
        nuc_pp, myo_pp = postprocess_masks(nuc_pred, myo_pred)

        # Save postprocessed masks for inspection
        Image.fromarray(nuc_pp * 255).save(out_pred_dir / f"{stem}_nuclei_pred.png")
        Image.fromarray(myo_pp * 255).save(out_pred_dir / f"{stem}_myotube_pred.png")

        # Compute biological metrics from the CLEAN masks
        m = compute_bio_metrics(nuc_pp, myo_pp)
        m["image"] = stem
        rows.append(m)

    # Save all metrics
    import csv
    if rows:
        keys = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print(f"[OK] Wrote predicted metrics for {len(rows)} images to {args.out_csv}")

if __name__ == "__main__":
    main()
