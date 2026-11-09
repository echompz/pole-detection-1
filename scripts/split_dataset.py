"""Split a flat folder of images (+ YOLO .txt labels) into train/val.

Works if the .txt label files sit next to the images, or in a separate folder (--labels).
Files are copied, so your originals are untouched.

Usage:
    python scripts/split_dataset.py /path/to/my_photos
    python scripts/split_dataset.py /path/to/images --labels /path/to/labels --val 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

p = argparse.ArgumentParser()
p.add_argument("images", help="folder containing all images")
p.add_argument("--labels", help="folder containing .txt labels (default: same as images)")
p.add_argument("--out", default="data/dataset")
p.add_argument("--val", type=float, default=0.2, help="fraction used for validation")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

src = Path(a.images)
lbl_dir = Path(a.labels) if a.labels else src
out = Path(a.out)

images = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
labeled = [f for f in images if (lbl_dir / f"{f.stem}.txt").exists()]
skipped = len(images) - len(labeled)
if not labeled:
    raise SystemExit(f"No images with matching .txt labels found in {src} / {lbl_dir}")

random.Random(a.seed).shuffle(labeled)
n_val = max(1, round(len(labeled) * a.val))
splits = {"val": labeled[:n_val], "train": labeled[n_val:]}

for name, files in splits.items():
    (out / "images" / name).mkdir(parents=True, exist_ok=True)
    (out / "labels" / name).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out / "images" / name / f.name)
        shutil.copy2(lbl_dir / f"{f.stem}.txt", out / "labels" / name / f"{f.stem}.txt")

print(f"train: {len(splits['train'])}  val: {len(splits['val'])}  skipped (no label): {skipped}")
