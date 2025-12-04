import pickle
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# ---------- load ----------
with open(r"C:\Users\styph\OneDrive\Desktop\Lab\myotube_models.pkl", "rb") as f:
    model_data = pickle.load(f)

models = model_data["models"]          # dict[target] -> trained RF
feature_names = model_data["feature_names"]  # list[str]

# ---------- feature extractor (same logic you used to train) ----------
def extract_features(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    feats = {}
    feats["mean_intensity"] = np.mean(gray)
    feats["std_intensity"]  = np.std(gray)
    feats["min_intensity"]  = np.min(gray)
    feats["max_intensity"]  = np.max(gray)

    for i, ch in enumerate(["blue", "green", "red"]):
        feats[f"{ch}_mean"] = np.mean(img[:, :, i])
        feats[f"{ch}_std"]  = np.std(img[:, :, i])

    for i, ch in enumerate(["hue", "saturation", "value"]):
        feats[f"{ch}_mean"] = np.mean(hsv[:, :, i])
        feats[f"{ch}_std"]  = np.std(hsv[:, :, i])

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    feats["laplacian_var"] = lap.var()

    edges = cv2.Canny(gray, 50, 150)
    feats["edge_density"] = np.sum(edges > 0) / edges.size

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats["num_contours"] = len(contours)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        feats["mean_contour_area"]  = np.mean(areas)
        feats["max_contour_area"]   = np.max(areas)
        feats["total_contour_area"] = np.sum(areas)
    else:
        feats["mean_contour_area"]  = 0
        feats["max_contour_area"]   = 0
        feats["total_contour_area"] = 0

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    feats["opening_mean"] = np.mean(opening)
    feats["closing_mean"] = np.mean(closing)

    # These were present in training; set to 0 for new images
    feats["passage"] = 0
    feats["well"]    = 0
    return feats

# ---------- predict ----------
def predict_all(img_path):
    feats = extract_features(img_path)
    # ensure same column order as training
    X = pd.DataFrame([feats], columns=feature_names).fillna(0)
    return {t: float(m.predict(X)[0]) for t, m in models.items()}

# ---------- run ----------
img_path = Path(r"C:\Users\styph\OneDrive\Desktop\Lab\original_images\Unsort_Stain_Well1_Image1.PNG")
preds = predict_all(img_path)
for k, v in preds.items():
    print(f"{k}: {v:.2f}")

