# Saudi Sign Language Recognition

Live recognition of Saudi Sign Language from a laptop webcam or a phone camera, running on a
laptop with no graphics card.

## Highlights

- **85.8% top-1 / 97.2% top-5 on signers the model has never seen** (KArSL-100, 100 signs,
  leave-one-signer-out). The published signer-independent result on this benchmark is 68.2%.
- **99.7%** on the official KArSL-100 test split.
- **2 MB ONNX model, under 2 ms per prediction** on a laptop CPU; trained on the same laptop.
- **Privacy by design:** the model sees only skeleton points, never the video.
- **Works in real conditions:** recovers hands that touch the chest, handles upright phone
  cameras and mirrored video, and ends a sign when the hands rest.
- **Browser app** for laptop and phone (phone joins with a QR code), with live Arabic and
  English results and a practice mode.

## How it works

```
camera  ->  MediaPipe (body, hands, 13 face points)  ->  normalise to the shoulders
        ->  32-frame landmark sequence              ->  Transformer (450k params)  ->  sign
```

The camera never reaches the neural network, so training data is tiny and inference is fast
on a CPU.

## Results

| Evaluation | Top-1 | Top-5 |
|---|---|---|
| New signers (leave-one-signer-out, mean of 3 folds) | **85.8%** | **97.2%** |
| Official KArSL-100 test split (2,400 clips) | **99.7%** | 100% |
| Published signer-independent result, same benchmark | 68.2% | |

| Stage, laptop CPU | Time |
|---|---|
| Classifier, ONNX Runtime | 1.6 ms |
| MediaPipe landmarks, 640x480 | 71 ms (about 15 FPS) |

Full experiment log: [`reports/results.md`](reports/results.md).

## Run it

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,app]"
python scripts/download_models.py
python -m app.server --run runs/final_v3_face --https     # browser app, laptop and phone
```

| Task | Command |
|---|---|
| Live webcam demo | `python -m ssl_recognition.infer --run runs/final_v3_face --onnx` |
| Train the deployment model | `python -m ssl_recognition.train --config configs/final_tf_all_official.yaml` |
| Export to ONNX | `python scripts/export_onnx.py --run runs/final_v3_face` |
| Benchmark | `python scripts/benchmark_cpu.py --run runs/final_v3_face` |
| Tests | `pytest -q && ruff check .` |

## Stack

Python, PyTorch, MediaPipe, ONNX Runtime, FastAPI, WebSockets, GitHub Actions.

## Data

[KArSL](https://hamzah-luqman.github.io/KArSL/) (KFUPM): 502 isolated Arabic signs, 3 signers.
KArSL-100 is signs 0071 to 0170.

## Licence

MIT for the code. Dataset licences belong to their owners.
