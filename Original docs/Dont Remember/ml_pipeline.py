import os
import re
import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class MyotubeAnalyzer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.feature_names = []

    # -------------------------
    # Utilities / Debug helpers
    # -------------------------
    def inspect_excel(self, excel_file: str):
        """Print columns and unique (passage, well) combos to help debugging."""
        xl_file = pd.ExcelFile(excel_file)
        all_data = [pd.read_excel(excel_file, sheet_name=s) for s in xl_file.sheet_names]
        df = pd.concat(all_data, ignore_index=True)
        raw_cols = df.columns.tolist()

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.rename(columns={
            '#_of_nuclei': 'num_nuclei',
            '#_of_myhc_positive_nuclei': 'num_myhc_positive',
            '%_of_myhc_positive_nucei': 'pct_myhc_positive',
            '#_of_nucei_that_have_fused_(myhc_positive-2_or_more)': 'num_fused',
            '#_of_myotubes': 'num_myotubes',
            'fusion_index_(%)': 'fusion_index'
        })

        # to_numeric to avoid crashes
        for col in ['passage_#', 'well_#']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("\n[inspect_excel] Raw Excel columns:\n", raw_cols)
        print("\n[inspect_excel] Normalized Excel columns:\n", df.columns.tolist())
        if {'passage_#', 'well_#'}.issubset(df.columns):
            combos = df[['passage_#', 'well_#']].dropna().drop_duplicates().sort_values(['passage_#', 'well_#'])
            print("\n[inspect_excel] Unique (passage_#, well_#) combos in Excel:")
            print(combos.to_string(index=False))
        else:
            print("\n[inspect_excel] Could not find passage_# and well_# columns after normalization.")

    # -------------------------
    # Feature extraction
    # -------------------------
    def extract_features(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        features = {
            "mean_intensity": np.mean(gray),
            "std_intensity": np.std(gray),
            "min_intensity": np.min(gray),
            "max_intensity": np.max(gray),
        }

        for i, channel in enumerate(["blue", "green", "red"]):
            features[f"{channel}_mean"] = np.mean(img[:, :, i])
            features[f"{channel}_std"] = np.std(img[:, :, i])

        for i, channel in enumerate(["hue", "saturation", "value"]):
            features[f"{channel}_mean"] = np.mean(hsv[:, :, i])
            features[f"{channel}_std"] = np.std(hsv[:, :, i])

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features["laplacian_var"] = laplacian.var()

        edges = cv2.Canny(gray, 50, 150)
        features["edge_density"] = np.sum(edges > 0) / edges.size

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features["num_contours"] = len(contours)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features["mean_contour_area"] = float(np.mean(areas))
            features["max_contour_area"] = float(np.max(areas))
            features["total_contour_area"] = float(np.sum(areas))
        else:
            features["mean_contour_area"] = 0.0
            features["max_contour_area"] = 0.0
            features["total_contour_area"] = 0.0

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        features["opening_mean"] = float(np.mean(opening))
        features["closing_mean"] = float(np.mean(closing))

        return features

    # -------------------------
    # File name parsing
    # -------------------------
    def parse_filename(self, filename):
        """Try several regexes to get passage & well.
        Returns (passage, well) where either may be None if not found."""
        f = os.path.basename(filename)

        # Try patterns that capture both passage and well
        patterns_both = [
            r"[pP](\d+).*?[wW]ell[_\s]?(\d+)",  # P1_Well2, p1well2
            r"[bB](\d+).*?[wW]ell[_\s]?(\d+)",  # B1_Well2
        ]
        for pat in patterns_both:
            m = re.search(pat, f)
            if m:
                return int(m.group(1)), int(m.group(2))

        # Passage only (e.g., B1 or P2)
        m_pass = re.search(r"[bBpP](\d+)", f)
        passage = int(m_pass.group(1)) if m_pass else None

        # Well only (e.g., Well2 or well_3)
        m_well = re.search(r"[wW]ell[_\s]?(\d+)", f)
        well = int(m_well.group(1)) if m_well else None

        return passage, well

    # -------------------------
    # Dataset preparation
    # -------------------------
    def prepare_dataset(self, image_dir, excel_file, allow_well_only_match=True):
        # Load + normalize excel
        xl_file = pd.ExcelFile(excel_file)
        all_data = [pd.read_excel(excel_file, sheet_name=s) for s in xl_file.sheet_names]
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.columns = combined_df.columns.str.strip().str.lower().str.replace(" ", "_")

        combined_df = combined_df.rename(columns={
            '#_of_nuclei': 'num_nuclei',
            '#_of_myhc_positive_nuclei': 'num_myhc_positive',
            '%_of_myhc_positive_nucei': 'pct_myhc_positive',
            '#_of_nucei_that_have_fused_(myhc_positive-2_or_more)': 'num_fused',
            '#_of_myotubes': 'num_myotubes',
            'fusion_index_(%)': 'fusion_index'
        })

        # Make passage/well numeric
        for col in ['passage_#', 'well_#']:
            if col not in combined_df.columns:
                raise KeyError(f"Expected column '{col}' not found in Excel after normalization.")
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        before_drop = len(combined_df)
        combined_df = combined_df.dropna(subset=['passage_#', 'well_#'])
        after_drop = len(combined_df)
        print(f"[prepare_dataset] Dropped {before_drop - after_drop} rows with NaN passage_#/well_#")

        combined_df['passage_#'] = combined_df['passage_#'].astype(int)
        combined_df['well_#'] = combined_df['well_#'].astype(int)

        print("[prepare_dataset] Excel unique (passage_#, well_#) combos:")
        print(combined_df[['passage_#', 'well_#']].drop_duplicates().sort_values(['passage_#', 'well_#']).to_string(index=False))

        # Collect features/targets
        features_list, targets_list = [], []
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"[prepare_dataset] Found {len(image_files)} image files in '{image_dir}'")

        unmatched = []
        matched = 0

        for img_file in image_files:
            passage, well = self.parse_filename(img_file)
            print(f"Checking image: {img_file} → Parsed (P{passage}, W{well})")

            if well is None:  # cannot match without well number
                print(f"❌ Skipping {img_file}: could not parse well number")
                unmatched.append((img_file, passage, well, 'no_well'))
                continue

            # First try exact (passage, well)
            match = combined_df[(combined_df['passage_#'] == (passage if passage is not None else -9999)) &
                                (combined_df['well_#'] == well)]

            # If nothing and allowed, try well-only unique match
            if match.empty and allow_well_only_match:
                well_only = combined_df[combined_df['well_#'] == well]
                if len(well_only) == 1:
                    print(f"ℹ️ Using well-only match for {img_file}: well={well}")
                    match = well_only
                    # overwrite passage with the one from excel
                    passage = int(match.iloc[0]['passage_#'])
                elif len(well_only) > 1:
                    print(f"❌ Ambiguous well-only match for {img_file}: {len(well_only)} rows. Skipping.")

            if match.empty:
                print(f"❌ No match in Excel for image: {img_file} (P{passage} W{well})")
                unmatched.append((img_file, passage, well, 'no_match'))
                continue

            img_path = os.path.join(image_dir, img_file)
            try:
                features = self.extract_features(img_path)
            except FileNotFoundError:
                print(f"❌ Image not found or unreadable: {img_file}")
                unmatched.append((img_file, passage, well, 'image_unreadable'))
                continue

            features.update({"passage": passage if passage is not None else 0, "well": well})
            features_list.append(features)

            row = match.iloc[0]
            targets_list.append({
                "num_nuclei": row.get("num_nuclei", 0),
                "num_myhc_positive": row.get("num_myhc_positive", 0),
                "pct_myhc_positive": row.get("pct_myhc_positive", 0),
                "num_fused": row.get("num_fused", 0),
                "num_myotubes": row.get("num_myotubes", 0),
                "fusion_index": row.get("fusion_index", 0)
            })

            matched += 1
            print(f"✅ Match found and processed: {img_file} → (P{passage}, W{well})")

        print(f"[prepare_dataset] Matched {matched} / {len(image_files)} images")
        if unmatched:
            print("[prepare_dataset] Unmatched / skipped images:")
            for u in unmatched:
                print("   ", u)

        features_df = pd.DataFrame(features_list)
        targets_df = pd.DataFrame(targets_list)
        self.feature_names = features_df.columns.tolist()
        return features_df, targets_df

    # -------------------------
    # Training / Prediction
    # -------------------------
    def train_models(self, features_df, targets_df):
        if features_df.empty or targets_df.empty:
            raise ValueError("No data to train on.")

        self.models = {}
        self.metrics = {}

        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_df, test_size=0.2, random_state=42
        )

        for target in targets_df.columns:
            print(f"Training model for {target}...")
            if y_train[target].isna().all() or (y_train[target] == 0).all():
                print(f"Skipping {target} - no valid data")
                continue

            valid_indices = ~y_train[target].isna()
            X_train_valid = X_train[valid_indices]
            y_train_valid = y_train[target][valid_indices]

            if len(X_train_valid) < 2:
                print(f"Skipping {target} - insufficient data")
                continue

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_valid, y_train_valid)

            valid_test_indices = ~y_test[target].isna()
            if valid_test_indices.sum() > 0:
                X_test_valid = X_test[valid_test_indices]
                y_test_valid = y_test[target][valid_test_indices]

                y_pred = model.predict(X_test_valid)
                mse = mean_squared_error(y_test_valid, y_pred)
                r2 = r2_score(y_test_valid, y_pred)
                self.metrics[target] = {"mse": mse, "r2": r2}
                print(f"{target} - MSE: {mse:.4f}, R2: {r2:.4f}")

            self.models[target] = model
        return self.models, self.metrics

    def predict(self, image_path):
        if not self.models:
            raise ValueError("Models not trained or loaded.")

        features = self.extract_features(image_path)
        features.update({"passage": 0, "well": 0})
        # align to training columns, fill missing with 0
        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        predictions = {t: float(m.predict(X)[0]) for t, m in self.models.items()}
        return predictions

    def save_models(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({
                "models": self.models,
                "feature_names": self.feature_names,
                "metrics": self.metrics
            }, f)

    def load_models(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]


if __name__ == "__main__":
    analyzer = MyotubeAnalyzer()

    image_dir = "."  # TODO: set your real images directory
    excel_file = "C:/Users/styph/OneDrive/Desktop/Lab/Passage # Experiment - James Martin.xlsx"

    # First: inspect excel to know what to match against
    analyzer.inspect_excel(excel_file)

    print("\nPreparing dataset...")
    features_df, targets_df = analyzer.prepare_dataset(image_dir, excel_file, allow_well_only_match=True)

    print(f"\nDataset prepared: {len(features_df)} samples, {len(features_df.columns)} features")
    print("Features:", features_df.columns.tolist())
    print("Targets:", targets_df.columns.tolist())

    if len(features_df) == 0:
        print("\n❗ No samples were matched. Please check: \n"
              "  • That image_dir points to the folder with your images\n"
              "  • The filename patterns (try renaming or extend parse_filename)\n"
              "  • The (passage_#, well_#) combos printed above from the Excel")
    else:
        print("\nTraining models...")
        models, metrics = analyzer.train_models(features_df, targets_df)

        print(f"\nTrained {len(models)} models")
        for target, metric in metrics.items():
            print(f"  {target}: MSE={metric['mse']:.4f}, R2={metric['r2']:.4f}")

        analyzer.save_models("myotube_model.pkl")
        print("\nModels saved to myotube_model.pkl")

        # Test prediction on one image (adjust path)
        test_image = os.path.join(image_dir, "B1_High_Well2_Image2.PNG")
        try:
            predictions = analyzer.predict(test_image)
            print(f"\nTest prediction for {test_image}:")
            for target, value in predictions.items():
                print(f"  {target}: {value:.2f}")
        except Exception as e:
            print(f"Prediction failed: {e}")




