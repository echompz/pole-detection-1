import io

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.core import Detection, count_materials, format_material_list, load_config
from app.detector import MockDetector
from app.discrepancy import compare

cfg = load_config()
DETS = [Detection("Insulator", 0.9, (10, 10, 30, 40)) for _ in range(6)] + [
    Detection("Tower", 0.8, (0, 0, 50, 300)),
    Detection("Insulator", 0.1, (5, 5, 9, 9)),  # below threshold, ignored
]


def test_count_and_format():
    counts = count_materials(DETS, cfg)
    assert counts == {"633444": 6, "TOWER": 1}
    assert format_material_list(counts) == ["6-633444", "1-TOWER"]


def test_compare():
    rows = {d.code: d.status for d in compare({"633444": 7, "600132": 2}, {"633444": 6})}
    assert rows == {"600132": "missing", "633444": "missing"}
    assert compare({"1": 2}, {"1": 3})[0].status == "over"


def test_api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "detector", MockDetector(DETS))
    monkeypatch.setattr(main, "RESULTS_FILE", tmp_path / "r.jsonl")
    buf = io.BytesIO()
    Image.new("RGB", (100, 300)).save(buf, "PNG")
    c = TestClient(main.app)
    r = c.post("/api/detect", files={"image": ("a.png", buf.getvalue(), "image/png")},
               data={"expected": '{"633444": 7}', "pole_id": "P1"})
    assert r.status_code == 200
    j = r.json()
    assert j["materials"] == ["6-633444", "1-TOWER"]
    assert j["total_insulators"] == 6
    assert j["has_discrepancy"] is True
    assert "633444" in c.get("/api/results.csv").text
    assert c.post("/api/detect", files={"image": ("a.txt", b"x", "text/plain")}).status_code == 400
