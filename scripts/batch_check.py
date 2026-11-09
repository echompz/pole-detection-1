"""Batch discrepancy check: run detector over a folder of photos and compare to expected counts.

expected.csv columns: image,code,qty   (image = file name)
Usage: python scripts/batch_check.py photos/ expected.csv --out report.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.core import count_materials, load_config
from app.detector import YoloDetector
from app.discrepancy import compare, load_expected_csv

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("expected")
p.add_argument("--weights", default="models/tic_small.pt")
p.add_argument("--out", default="report.csv")
a = p.parse_args()

cfg = load_config()
det = YoloDetector(a.weights, cfg)
expected = load_expected_csv(Path(a.expected))

bad = 0
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "code", "expected", "detected", "delta", "status"])
    for name, exp in expected.items():
        path = Path(a.folder) / name
        if not path.exists():
            print(f"missing file: {name}")
            continue
        counts = count_materials(det.detect(Image.open(path)), cfg)
        for d in compare(exp, counts):
            w.writerow([name, d.code, d.expected, d.detected, d.delta, d.status])
            bad += d.status != "match"
print(f"Wrote {a.out}; {bad} discrepancy rows")
