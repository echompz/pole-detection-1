"""Compare detected material counts against expected quantities."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Discrepancy:
    code: str
    expected: int
    detected: int

    @property
    def delta(self) -> int:
        return self.detected - self.expected

    @property
    def status(self) -> str:
        if self.delta == 0:
            return "match"
        return "over" if self.delta > 0 else "missing"


def compare(expected: dict[str, int], detected: dict[str, int]) -> list[Discrepancy]:
    """Return one row per material code present in either dict."""
    codes = sorted(set(expected) | set(detected))
    return [Discrepancy(c, expected.get(c, 0), detected.get(c, 0)) for c in codes]


def load_expected_csv(path: Path) -> dict[str, dict[str, int]]:
    """Read CSV with columns: image,code,qty -> {image: {code: qty}}."""
    out: dict[str, dict[str, int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            img = row["image"].strip()
            out.setdefault(img, {})[row["code"].strip()] = (
                out.get(img, {}).get(row["code"].strip(), 0) + int(row["qty"])
            )
    return out
