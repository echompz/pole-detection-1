# Line Materials Visual Inspection

Detects and counts line materials (insulators, poles, crossarms…) in a photo, works from a phone browser, and checks counts against expected quantities.

Output format matches your sample: `qty-materialcode`, e.g. `6-633444`.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://<your-computer-ip>:8000` on a phone (same Wi-Fi). The photo input opens the rear camera. For camera use outside your LAN, serve over HTTPS.

Without trained weights the app runs in demo mode (detects nothing). Put your weights at `models/tic_small.pt` (or set `INSULATOR_WEIGHTS`).

## Train on your pictures

1. Label images in YOLO format (Roboflow, CVAT, or Label Studio). Class names must match the keys in `config/materials.yaml` (`pin_insulator`, `pole`, `crossarm`, `crossarm_brace`).
2. Create `data/dataset.yaml` (`path`, `train`, `val`, `names`).
3. `python scripts/train.py data/dataset.yaml --epochs 100` → writes `models/tic_small.pt`.

Edit `config/materials.yaml` to map classes to your real material codes. The codes there come from the sample screenshot and are placeholders.

## Discrepancy checks

- **Single photo:** enter expected JSON in the UI, e.g. `{"633444": 6}`, or send it as the `expected` form field to `POST /api/detect`.
- **Batch:** `python scripts/batch_check.py photos/ expected.csv --out report.csv`, where `expected.csv` has columns `image,code,qty`. Status is `match`, `missing` or `over`.

## Sending the output

- Each result is saved to `data/results.jsonl`. `GET /api/results.csv` exports them.
- Set `RESULT_WEBHOOK_URL` to have every result POSTed as JSON to another system.

## Test

```bash
pytest
```

## Notes

- Accuracy depends entirely on your labeled data. Hold out a validation set and review the per-class results from training.
- The unit tests use a mock detector. The YOLO path was not run, because no weights exist yet.
