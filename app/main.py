"""FastAPI backend: detect/count materials from a photo, check discrepancies, export results."""
from __future__ import annotations

import base64
import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .core import count_materials, format_material_list, load_config
from .detector import MockDetector, YoloDetector, draw_detections
from .discrepancy import compare

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "data" / "results.jsonl"
WEIGHTS = os.environ.get("INSULATOR_WEIGHTS", str(ROOT / "models" / "tic_small.pt"))
WEBHOOK_URL = os.environ.get("RESULT_WEBHOOK_URL")  # optional: POST each result here

app = FastAPI(title="Line Materials Visual Inspection")
cfg = load_config()


def _build_detector():
    if Path(WEIGHTS).exists():
        return YoloDetector(WEIGHTS, cfg)
    print(f"[warn] weights not found at {WEIGHTS}; detector returns nothing (demo mode)")
    return MockDetector()


detector = _build_detector()


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": isinstance(detector, MockDetector)}


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    expected: str | None = Form(None, description='JSON, e.g. {"633444": 6}'),
    pole_id: str | None = Form(None),
):
    try:
        img = Image.open(io.BytesIO(await image.read()))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not read image")

    detections = detector.detect(img)
    counts = count_materials(detections, cfg)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pole_id": pole_id,
        "image": image.filename,
        "counts": counts,
        "materials": format_material_list(counts),
        "total_insulators": sum(
            n for c, n in counts.items()
            if c in {str(m.get("code")) for k, m in cfg.materials.items() if "insulator" in k.lower()}
        ),
    }

    if expected:
        try:
            exp = {str(k): int(v) for k, v in json.loads(expected).items()}
        except (ValueError, AttributeError):
            raise HTTPException(400, "expected must be a JSON object of code -> qty")
        result["discrepancies"] = [
            {"code": d.code, "expected": d.expected, "detected": d.detected,
             "delta": d.delta, "status": d.status}
            for d in compare(exp, counts)
        ]
        result["has_discrepancy"] = any(d["status"] != "match" for d in result["discrepancies"])

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    if WEBHOOK_URL:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(WEBHOOK_URL, json=result)
        except httpx.HTTPError as e:
            result["webhook_error"] = str(e)

    buf = io.BytesIO()
    draw_detections(img, detections, cfg).save(buf, format="JPEG", quality=85)
    result["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return result


@app.get("/api/results.csv", response_class=PlainTextResponse)
def results_csv():
    """Export all past results (one row per image/material code)."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["timestamp", "pole_id", "image", "code", "qty"])
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            r = json.loads(line)
            for code, qty in r["counts"].items():
                w.writerow([r["timestamp"], r["pole_id"], r["image"], code, qty])
    return out.getvalue()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
