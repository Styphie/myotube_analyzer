
"""
metrics_eval.py
---------------
Compare predicted biological metrics (from your model run) to manual ground truth.

Ground truth sources:
  A) Excel "Passage.xlsx"  (--excel + --passage)
  B) CSV with columns: "Well #", gt_*                 (--gt_csv)

Predicted input: --pred_csv runs/<timestamp>/metrics_val.csv
Outputs: eval_summary.csv, manual_vs_pred.csv, scatter_<metric>.png
"""
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


PRED_METRICS = [
    ("total_nuclei", "gt_total_nuclei", "total_nuclei"),
    ("myHC_positive_nuclei", "gt_myHC_positive_nuclei", "myHC_positive_nuclei"),
    ("myotube_count", "gt_myotube_count", "myotube_count"),
    ("fusion_index", "gt_fusion_index", "fusion_index"),
    ("nuclei_fused", "gt_nuclei_fused", "nuclei_fused"),
    ("avg_nuclei_per_myotube", "gt_avg_nuclei_per_myotube", "avg_nuclei_per_myotube"),
    ("myHC_positive_percentage", "gt_myHC_positive_percentage", "myHC_positive_percentage"),
]

def well_from_name(name: str):
    m = re.search(r'well\s*(\d+)', str(name), flags=re.I)
    return int(m.group(1)) if m else None

def load_manual_from_excel(xlsx_path: Path, passage: int) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if "Passage #" in df.columns:
            frames.append(df)
    if not frames:
        raise RuntimeError("No sheets with 'Passage #' found in Excel.")
    manual_all = pd.concat(frames, ignore_index=True)

    # 🔹 If passage <= 0 → use ALL passages
    if passage is not None and passage > 0:
        manual_p = manual_all[manual_all["Passage #"] == passage].copy()
        if manual_p.empty:
            raise RuntimeError(f"No rows for Passage #{passage}.")
    else:
        manual_p = manual_all.copy()

    rename_map = {
        "# of Nuclei": "gt_total_nuclei",
        "# of MyHC Positive Nuclei": "gt_myHC_positive_nuclei",
        "# of MyHC Positive Nuclei ": "gt_myHC_positive_nuclei",
        "% of MyHC Positive Nucei": "gt_myHC_positive_percentage",
        "% of MyHC Positive Nuclei": "gt_myHC_positive_percentage",
        "# of Nucei that have Fused (MyHC positive-2 or more)": "gt_nuclei_fused",
        "# of Nuclei that have Fused (MyHC positive-2 or more)": "gt_nuclei_fused",
        "# of Myotubes": "gt_myotube_count",
        "Average # of Nuclei per Myotube": "gt_avg_nuclei_per_myotube",
        "Fusion Index (%)": "gt_fusion_index",
        "Fusion Index (%) ": "gt_fusion_index",
        "Well #": "Well #",
    }
    manual_p = manual_p.rename(columns=rename_map)
    keep = ["Well #","gt_total_nuclei","gt_myHC_positive_nuclei","gt_myHC_positive_percentage",
            "gt_nuclei_fused","gt_myotube_count","gt_avg_nuclei_per_myotube","gt_fusion_index"]
    for k in keep:
        if k not in manual_p.columns:
            manual_p[k] = np.nan
    manual_p = manual_p[keep].reset_index(drop=True)
    return manual_p

def load_manual_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Well #" not in df.columns:
        raise RuntimeError("Ground-truth CSV must include a 'Well #' column.")
    return df

def scatter_plot(gt, pr, title, xlabel, ylabel, save_path: Path):
    plt.figure(figsize=(5,5))
    plt.scatter(gt, pr)
    lims = [min(min(gt), min(pr)), max(max(gt), max(pr))]
    plt.plot(lims, lims)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--excel", default=None)
    ap.add_argument("--passage", type=int, default=None)
    ap.add_argument("--gt_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    if "image" not in pred.columns:
        raise RuntimeError("pred_csv must contain an 'image' column.")
    pred["Well #"] = pred["image"].apply(well_from_name)

    if args.gt_csv:
        manual = load_manual_from_csv(Path(args.gt_csv))
    else:
        if not args.excel:
            raise RuntimeError("Provide either --gt_csv OR --excel (and optional --passage).")
        # args.passage can be None or <=0 → means "all passages"
        manual = load_manual_from_excel(Path(args.excel), args.passage if args.passage is not None else 0)

    merged = manual.merge(pred, on="Well #", how="inner")
    if merged.empty:
        raise RuntimeError("Merged table is empty; check file names and 'Well #' mapping.")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pred_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, gt_col, pr_col in PRED_METRICS:
        if gt_col not in merged.columns or pr_col not in merged.columns:
            continue
        gt = pd.to_numeric(merged[gt_col], errors="coerce").dropna()
        pr = pd.to_numeric(merged.loc[gt.index, pr_col], errors="coerce")
        if len(gt)==0: continue
        mae = mean_absolute_error(gt, pr)
        r2 = r2_score(gt, pr)
        rows.append({"metric": label, "MAE": round(mae,3), "R2": round(r2,3)})
        scatter_plot(gt.values, pr.values,
                     f"{label}: manual vs predicted", "Manual", "Predicted",
                     out_dir/f"scatter_{label}.png")

    eval_df = pd.DataFrame(rows).sort_values("metric")
    merged.to_csv(out_dir/"manual_vs_pred.csv", index=False)
    eval_df.to_csv(out_dir/"eval_summary.csv", index=False)
    print("[OK] Wrote:", out_dir/"manual_vs_pred.csv", "and", out_dir/"eval_summary.csv")

if __name__ == "__main__":
    main()
