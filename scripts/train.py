"""Train YOLO on your labeled insulator photos.

Dataset layout (YOLO format) described by data/dataset.yaml:
    path: data/dataset
    train: images/train
    val: images/val
    names: {0: pin_insulator, 1: pole, 2: crossarm, 3: crossarm_brace}
Class names must match the keys in config/materials.yaml.

Usage: python scripts/train.py data/dataset.yaml --epochs 100
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--base", default="yolov8n.pt")
p.add_argument("--device", default="mps")
a = p.parse_args()

model = YOLO(a.base)
res = model.train(data=a.data, epochs=a.epochs, imgsz=960, device=a.device, project="runs", name="insulator")
Path("models").mkdir(exist_ok=True)
shutil.copy(Path(res.save_dir) / "weights" / "best.pt", "models/best.pt")
print("Saved models/best.pt")
