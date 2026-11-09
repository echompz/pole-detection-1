"""Convert Pascal VOC XML labels to YOLO .txt labels.

Class ids follow the order in --classes (default: insulator).
Usage: python scripts/voc_to_yolo.py raw/labels --out raw/labels_yolo --classes insulator
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("labels")
p.add_argument("--out", required=True)
p.add_argument("--classes", nargs="+", default=["insulator"])
a = p.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ids = {n: i for i, n in enumerate(a.classes)}
n_files = n_boxes = n_unknown = 0

for xml in sorted(Path(a.labels).glob("*.xml")):
    root = ET.parse(xml).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    lines = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        if name not in ids:
            n_unknown += 1
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, x1), min(w, x2)
        y1, y2 = max(0.0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{ids[name]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    (out / f"{xml.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    n_files += 1
    n_boxes += len(lines)

print(f"{n_files} files, {n_boxes} boxes, {n_unknown} skipped (unknown class)")
