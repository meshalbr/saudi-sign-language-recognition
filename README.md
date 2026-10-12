# Saudi Sign Language Recognition

Real-time recognition of isolated Saudi Sign Language signs from a webcam, running on a
low-end laptop CPU (no CUDA). The camera never reaches the model: MediaPipe Holistic turns each
frame into 155 body-normalised landmark features, and a small bidirectional GRU classifies the
landmark sequence. That keeps training data tiny, inference fast, and the signer's face out of
the pipeline.

> Status: week 0. Pipeline, recorder, trainer, ONNX export and live demo are in place and tested.
> Dataset access (KArSL, KFUPM) requested. Results table will appear below as experiments land.

## Pipeline

```
webcam / video  ->  MediaPipe Holistic (pose + hands)  ->  155-d features per frame
                ->  resample to 32 frames  ->  BiGRU classifier  ->  sign + confidence
```

Feature vector per frame: 9 upper-body pose points, 21 left-hand points, 21 right-hand points
(all x, y, z, centred on the shoulder midpoint and scaled by shoulder width), plus two hand-present
flags. Augmentation: mirror (swapping hands), temporal crop, scale, shift, jitter.

## Setup

```
conda create -p ./.env python=3.11 -y          # any 3.10-3.12 interpreter works
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python scripts/download_models.py             # MediaPipe holistic model, 14 MB
```

## Use

| Step | Command |
|---|---|
| Record your own clips | `python -m ssl_recognition.record --label hello --signer meshal --clips 10` |
| Extract landmarks from a video dataset | `python scripts/extract_landmarks.py --videos data/videos --signer-from-name` |
| Train | `python -m ssl_recognition.train --config configs/baseline.yaml --test-signer s3` |
| Export to ONNX | `python scripts/export_onnx.py --run runs/baseline` |
| Live demo | `python -m ssl_recognition.infer --run runs/baseline --onnx` |
| Benchmark this machine | `python scripts/benchmark_cpu.py --run runs/baseline` |
| Tests and lint | `pytest -q && ruff check .` |

Data layout: `data/landmarks/<label>/<signer>_<id>.npz`, one small file per clip.
`--test-signer` holds out one signer entirely, which is the honest evaluation for sign language
(random splits leak signer style into the test set).

## Results

| Date | Data | Classes | Split | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|---|
| | | | | | | see `reports/results.md` |

## Edge benchmark

Latency on the development laptop (AMD Ryzen 5 3500U, 4 cores, 7 GB RAM, Radeon Vega 8, no CUDA)
will be reported here: MediaPipe holistic per frame, classifier in PyTorch vs ONNX Runtime,
CPU vs DirectML, FP32 vs INT8.

## Roadmap

1. Baseline GRU on 50 signs from own recordings and KArSL skeleton data
2. Scale to all 502 KArSL signs, signer-independent evaluation
3. ONNX export, INT8 quantisation, DirectML vs CPU benchmark
4. Live demo with sign-boundary detection
5. Stretch: continuous recognition (ArabSign)

## Data

- [KArSL](https://faculty.kfupm.edu.sa/ICS/hluqman/datasets.html): 502 signs, 3 signers, 75,300 samples (KFUPM, research licence, access on request)
- Own recordings through `ssl_recognition.record`

## Licence

MIT. Dataset licences are separate and belong to their owners.
