"""Object detectors. YoloDetector uses your trained weights; MockDetector is for tests/demo."""
from __future__ import annotations

from PIL import Image, ImageDraw

from .core import Config, Detection


class YoloDetector:
    def __init__(self, weights: str, cfg: Config):
        from ultralytics import YOLO  # imported lazily so tests don't need it

        self.model = YOLO(weights)
        self.cfg = cfg

    def detect(self, image: Image.Image) -> list[Detection]:
        result = self.model.predict(image, conf=self.cfg.min_confidence, verbose=False)[0]
        names = result.names
        return [
            Detection(names[int(b.cls)], float(b.conf), tuple(map(float, b.xyxy[0].tolist())))
            for b in result.boxes
        ]


class MockDetector:
    """Returns fixed detections; lets the API run without trained weights."""

    def __init__(self, detections: list[Detection] | None = None):
        self.detections = detections or []

    def detect(self, image: Image.Image) -> list[Detection]:
        return list(self.detections)


def draw_detections(image: Image.Image, detections: list[Detection], cfg: Config) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for det in detections:
        if det.confidence < cfg.min_confidence:
            continue
        d.rectangle(det.box, outline=(255, 165, 0), width=3)
        d.text(
            (det.box[0] + 2, max(det.box[1] - 12, 0)),
            f"{cfg.code_for(det.label)} {det.confidence:.2f}",
            fill=(255, 165, 0),
        )
    return img
