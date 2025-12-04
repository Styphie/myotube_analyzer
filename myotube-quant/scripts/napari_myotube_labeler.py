
"""
napari_myotube_labeler.py

Interactively annotate nuclei + myotubes in napari and save masks.

Usage (from project root):
  conda activate myotubes
  python scripts/napari_myotube_labeler.py --data_root "C:/Users/you/Desktop/Lab/myotube-quant/data" [--start_stem "P12 well 2"]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import napari
from skimage import io as skio


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Load RGB image as numpy array."""
    img = skio.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def load_mask(path: Path, shape) -> np.ndarray:
    """Load mask if exists, otherwise return empty mask."""
    if path.exists():
        m = skio.imread(str(path))
        if m.ndim > 2:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8)
        # resize if needed
        if m.shape != shape[:2]:
            m = np.array(Image.fromarray(m).resize((shape[1], shape[0]), Image.NEAREST))
        return m
    else:
        return np.zeros(shape[:2], dtype=np.uint8)


def save_mask(mask: np.ndarray, path: Path):
    """Save binary mask as 0/255 PNG/TIF."""
    ensure_dir(path.parent)
    arr = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to data/ folder")
    ap.add_argument("--start_stem", default=None, help="Optional image stem to start from, e.g. 'P12 well 2'")
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root / "images"
    nuc_dir = root / "masks" / "nuclei"
    myo_dir = root / "masks" / "myotubes"

    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # pick starting index
    idx = 0
    if args.start_stem is not None:
        for i, p in enumerate(images):
            if p.stem == args.start_stem:
                idx = i
                break

    viewer = napari.Viewer()
    state = {"idx": idx, "images": images}

    def load_current():
        """Load current image + masks into the viewer."""
        viewer.layers.clear()
        ip = state["images"][state["idx"]]
        img = load_image(ip)
        stem = ip.stem

        nuc_path = nuc_dir / f"{stem}.tif"
        myo_path = myo_dir / f"{stem}.tif"

        nuc_mask = load_mask(nuc_path, img.shape)
        myo_mask = load_mask(myo_path, img.shape)

        viewer.add_image(img, name="image", rgb=True)
        viewer.add_labels(nuc_mask, name="nuclei", opacity=0.7)
        viewer.add_labels(myo_mask, name="myotubes", opacity=0.5)

        viewer.title = f"{stem}   [{state['idx']+1}/{len(state['images'])}]"
        print(f"[LOADED] {stem}")

    def save_current():
        """Save nuclei + myotube masks for the current image."""
        ip = state["images"][state["idx"]]
        stem = ip.stem

        nuc_layer = viewer.layers["nuclei"].data
        myo_layer = viewer.layers["myotubes"].data

        nuc_path = nuc_dir / f"{stem}.tif"
        myo_path = myo_dir / f"{stem}.tif"

        save_mask(nuc_layer, nuc_path)
        save_mask(myo_layer, myo_path)

        print(f"[SAVED] {stem} -> {nuc_path.name}, {myo_path.name}")

    @viewer.bind_key("s")
    def save_and_print(viewer):
        """Press 's' to save current masks."""
        save_current()

    @viewer.bind_key("n")
    def next_image(viewer):
        """Press 'n' for next image (saving first)."""
        save_current()
        state["idx"] = (state["idx"] + 1) % len(state["images"])
        load_current()

    @viewer.bind_key("p")
    def prev_image(viewer):
        """Press 'p' for previous image (saving first)."""
        save_current()
        state["idx"] = (state["idx"] - 1) % len(state["images"])
        load_current()

    @viewer.bind_key("q")
    def quit_and_save(viewer):
        """Press 'q' to save and close."""
        save_current()
        viewer.close()

    # initial load
    load_current()
    napari.run()


if __name__ == "__main__":
    main()
