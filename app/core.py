"""Shared types, config loading, and material counting."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "materials.yaml"


@dataclass
class Detection:
    label: str  # detector class name
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Config:
    min_confidence: float = 0.35
    materials: dict[str, dict] = field(default_factory=dict)  # class name -> {code, description}

    def code_for(self, label: str) -> str:
        return str(self.materials.get(label, {}).get("code", label))


def load_config(path: Path = CONFIG_PATH) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        min_confidence=float(data.get("min_confidence", 0.35)),
        materials=data.get("materials", {}),
    )


def count_materials(detections: list[Detection], cfg: Config) -> dict[str, int]:
    """Count detections per material code, ignoring those below min_confidence."""
    counts = Counter(
        cfg.code_for(d.label) for d in detections if d.confidence >= cfg.min_confidence
    )
    return dict(sorted(counts.items()))


def format_material_list(counts: dict[str, int]) -> list[str]:
    """Format as 'qty-code' lines, e.g. '6-633444' (matches the sample output)."""
    return [f"{qty}-{code}" for code, qty in counts.items()]
