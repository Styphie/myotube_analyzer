"""
train_myotube_nuclei_unet.py
----------------------------
Trains a 2-head U-Net that takes (red, blue) channels and predicts
two binary masks: (nuclei, myotubes). Saves best model and validation
predictions + biological metrics.

DATA LAYOUT:
  data_root/
    images/
    masks_nuclei/
    masks_myotube/

Example (Windows):
  python train_myotube_nuclei_unet.py --data_root "C:/Users/you/.../myotube-quant/data" --epochs 50
"""

import os
import argparse, time, random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk

def postprocess_masks(nuc_mask, myo_mask,
                      min_nuc_area=20,
                      min_myo_area=500,
                      myo_close_radius=3):
    # nuclei: remove tiny specks
    nuc_lab, _ = ndi.label(nuc_mask.astype(bool))
    nuc_clean = remove_small_objects(nuc_lab, min_size=min_nuc_area)
    nuc_clean = (nuc_clean > 0).astype(np.uint8)

    # myotubes: close gaps, remove tiny blobs
    myo_bin = myo_mask.astype(bool)
    selem = disk(myo_close_radius)
    myo_closed = binary_closing(myo_bin, selem)
    myo_opened = binary_opening(myo_closed, selem)
    myo_lab, _ = ndi.label(myo_opened)
    myo_clean = remove_small_objects(myo_lab, min_size=min_myo_area)
    myo_clean = (myo_clean > 0).astype(np.uint8)

    return nuc_clean, myo_clean


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_rgb(p: Path): return np.array(Image.open(p).convert("RGB"))
def read_mask(p: Path): return (np.array(Image.open(p).convert("L")) > 0).astype(np.uint8)
def save_mask(arr, p: Path):
    Image.fromarray((arr>0).astype(np.uint8)*255).save(p)

class MyoDataset(Dataset):
    def __init__(self, root, image_size=512, augment=True):
        self.root = Path(root)

        #  Use paths *relative* to data_root
        # data_root/
        #   images/
        #   masks/
        #     nuclei/
        #     myotubes/
        self.img_dir = self.root / "images"
        self.nuc_dir = self.root / "masks" / "Nuclei_m"
        self.myo_dir = self.root / "masks" / "Myotubes_m"

        self.images = sorted(
            [p for p in self.img_dir.glob("*")
             if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]]
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        # keep only images that have both nuclei and myotube masks
        valid = []
        for p in self.images:
            stem = p.stem
            nuc = self.nuc_dir / f"{stem}.tif"
            if not nuc.exists():
                nuc = self.nuc_dir / f"{stem}.png"

            myo = self.myo_dir / f"{stem}.tif"
            if not myo.exists():
                myo = self.myo_dir / f"{stem}.png"

            if nuc.exists() and myo.exists():
                valid.append(p)

        self.images = valid
        if not self.images:
            raise FileNotFoundError("No images with both nuclei and myotube masks found.")

        self.size = image_size
        self.augment = augment

    def __len__(self): return len(self.images)

    def _aug(self, img, n, m):
        if np.random.rand() < 0.5:
            img = np.flip(img, 1); n = np.flip(n, 1); m = np.flip(m, 1)
        if np.random.rand() < 0.5:
            img = np.flip(img, 0); n = np.flip(n, 0); m = np.flip(m, 0)
        k = np.random.randint(0,4)
        if k: img = np.rot90(img, k); n = np.rot90(n, k); m = np.rot90(m, k)
        return img, n, m

    def __getitem__(self, idx):
        ip = self.images[idx]; stem = ip.stem
        rgb = read_rgb(ip).astype(np.float32)/255.0
        red = rgb[...,0]; blue = rgb[...,2]
        x = np.stack([red, blue], axis=0)

        nuc_p = (self.nuc_dir/f"{stem}.tif");  myo_p = (self.myo_dir/f"{stem}.tif")
        if not nuc_p.exists(): nuc_p = (self.nuc_dir/f"{stem}.png")
        if not myo_p.exists(): myo_p = (self.myo_dir/f"{stem}.png")
        if not nuc_p.exists() or not myo_p.exists():
            raise FileNotFoundError(f"Missing masks for {stem}")
        y_n = read_mask(nuc_p); y_m = read_mask(myo_p)

        H,W = self.size, self.size
        x = np.stack([
            np.array(Image.fromarray(x[0]).resize((W,H), Image.BILINEAR)),
            np.array(Image.fromarray(x[1]).resize((W,H), Image.BILINEAR)),
        ], axis=0)
        y_n = np.array(Image.fromarray(y_n).resize((W,H), Image.NEAREST))
        y_m = np.array(Image.fromarray(y_m).resize((W,H), Image.NEAREST))

        if self.augment:
            fake3 = np.stack([x[0], x[1], np.zeros_like(x[0])], axis=-1)
            fake3, y_n, y_m = self._aug(fake3, y_n, y_m)
            x = np.stack([fake3[...,0], fake3[...,1]], axis=0)

        y = np.stack([y_n, y_m], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), stem

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base);     self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2);    self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4);  self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8);  self.p4 = nn.MaxPool2d(2)
        self.bn = DoubleConv(base*8, base*16)
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.du4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2);  self.du3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2);  self.du2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2);    self.du1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        b  = self.bn(p4)
        x = self.u4(b); x = torch.cat([x,d4],1); x = self.du4(x)
        x = self.u3(x); x = torch.cat([x,d3],1); x = self.du3(x)
        x = self.u2(x); x = torch.cat([x,d2],1); x = self.du2(x)
        x = self.u1(x); x = torch.cat([x,d1],1); x = self.du1(x)
        return self.out(x)

class BCEDiceLoss(nn.Module):
    def __init__(self, w=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(); self.w = w
    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs*target).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = 1 - (2*inter + 1e-6)/(union + 1e-6)
        return self.w*bce + (1-self.w)*dice.mean()

@torch.no_grad()
def dice_coef(probs, target, thr=0.5):
    pred = (probs>thr).float()
    inter = (pred*target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2*inter + 1e-6)/(union + 1e-6)
    return dice.mean(dim=0)

import scipy.ndimage as ndi
from skimage import measure
def label_cc(mask):
    lab, _ = ndi.label(mask.astype(np.uint8)); return lab

def compute_bio_metrics(nuc_mask, myo_mask, min_overlap_frac=0.1):
    nuc_lab = label_cc(nuc_mask)
    myo_lab = label_cc(myo_mask)
    total = int(nuc_lab.max())

    myo_bin = myo_lab > 0
    pos = 0
    nm = {}  # myotube_id -> list of nucleus ids

    for prop in measure.regionprops(nuc_lab):
        coords = prop.coords
        nid = prop.label

        # Overlapping myotube IDs
        ids = myo_lab[coords[:, 0], coords[:, 1]]
        ids = ids[ids > 0]
        if ids.size == 0:
            continue

        # Most overlapped myotube
        unique, counts = np.unique(ids, return_counts=True)
        mt = int(unique[np.argmax(counts)])

        frac = counts.max() / len(coords)  # fraction of nucleus pixels inside that myotube
        if frac >= min_overlap_frac:
            pos += 1
            nm.setdefault(mt, []).append(nid)

    per = [len(v) for v in nm.values()]
    fused = sum(n for n in per if n >= 2)
    fi = 100.0 * fused / total if total else 0.0
    pct = 100.0 * pos / total if total else 0.0
    avg = float(np.mean(per)) if per else 0.0

    return {
        "total_nuclei": total,
        "myHC_positive_nuclei": int(pos),
        "myHC_positive_percentage": round(pct, 2),
        "nuclei_fused": int(fused),
        "myotube_count": int(len(per)),
        "avg_nuclei_per_myotube": round(avg, 2),
        "fusion_index": round(fi, 2),
    }

def train_one_epoch(model, loader, opt, device, loss_fn):
    model.train(); tot=0.0
    for x,y,_ in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); logits = model(x)
        loss = loss_fn(logits, y); loss.backward(); opt.step()
        tot += loss.item()*x.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, out_dir: Path):
    model.eval()
    ensure_dir(out_dir)
    dices = []
    rows = []

    for x, y, stem in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu()

        # Dice per channel
        d = dice_coef(probs, y).numpy()
        dices.append(d)

        # Threshold predictions to uint8 masks
        pr = (probs.numpy()[0] > 0.5).astype(np.uint8)

        # Save prediction masks
        save_mask(pr[0], out_dir / f"{stem[0]}_nuclei_pred.png")
        save_mask(pr[1], out_dir / f"{stem[0]}_myotube_pred.png")

        # Post-process masks to clean up junk / fragments
        nuc_pp, myo_pp = postprocess_masks(pr[0], pr[1])

        # Compute biological metrics from prediction
        m = compute_bio_metrics(nuc_pp, myo_pp)
        m["image"] = stem[0]
        rows.append(m)

    dices = np.array(dices)
    return rows, float(dices[:, 0].mean()), float(dices[:, 1].mean())

def main():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path("runs")/time.strftime("%Y%m%d_%H%M%S"); ensure_dir(run_dir)

    ds = MyoDataset(a.data_root, image_size=a.image_size, augment=True)
    n_val = max(1, int(len(ds)*a.val_split)); n_train = len(ds)-n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(a.seed))
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = UNet(in_ch=2, out_ch=2, base=32).to(device)
    loss_fn = BCEDiceLoss(0.5)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best=-1.0
    for ep in range(1, a.epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_fn)
        rows, d_nuc, d_myo = validate(model, val_loader, device, run_dir/"val_predictions")
        score = (d_nuc+d_myo)/2.0
        print(f"Epoch {ep:03d} | loss {tr:.4f} | dice_nuc {d_nuc:.3f} | dice_myo {d_myo:.3f}")
        if score>best:
            best=score
            torch.save({"model": model.state_dict(), "args": vars(a)}, run_dir/"model_best.pt")
            import csv
            keys = list(rows[0].keys()) if rows else ["image"]
            with open(run_dir/"metrics_val.csv","w",newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("Done. Best:", best, "Run:", str(run_dir))

if __name__ == "__main__":
    main()
