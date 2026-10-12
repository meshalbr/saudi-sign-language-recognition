# Saudi Sign Language Recognition

Real-time recognition of Saudi Sign Language signs from an ordinary webcam, running on a
low-end laptop with no graphics card.

The camera never reaches the neural network. MediaPipe Holistic turns each frame into 155
body-normalised landmark numbers, and a small bidirectional GRU classifies the sequence of
frames into a sign. That keeps the training data tiny, inference fast on a CPU, and the
signer's face out of the pipeline.

> Status: day 1. Pipeline, recorder, trainer, ONNX export and live demo are in place and
> tested (11 tests, CI green). First baseline: **99.0% top-1 on KArSL-100 (100 signs, official
> split, signer 01)** with a 566k-parameter BiGRU trained on the laptop CPU. Signers 02 and 03
> are being extracted for the signer-independent evaluation, which is the number that matters.

## Why this matters

Deaf and hard-of-hearing people in Saudi Arabia sign in Saudi Sign Language. Very few hearing
people understand it, and almost no software does. A recogniser that works on cheap hardware
is the first building block for tools that the community can actually use:

- **Translation aids** at service counters, clinics and classrooms, where no interpreter is
  present and no GPU server is available.
- **Learning apps** that give a hearing learner instant feedback on their signing.
- **Research on Saudi signs specifically.** Most published work covers American or
  generic Arabic sign language. The KArSL and Isharah datasets from KFUPM and IAU make
  Saudi-specific work possible, and this repo is a reproducible baseline on them.
- **Privacy by design.** The model only ever sees skeleton points, never the video, so a
  deployment does not have to store or transmit anyone's face.

The long-term goal is continuous recognition of whole signed sentences on a phone.

## How it works

```
webcam / video  ->  MediaPipe Holistic (pose + hands)  ->  155 numbers per frame
                ->  normalise to the shoulders        ->  resample to 32 frames
                ->  BiGRU classifier                  ->  sign + confidence
```

1. **Landmarks.** MediaPipe finds 33 body points and 21 points per hand in every frame. The
   features keep 9 upper-body points (nose, shoulders, elbows, wrists, hips) and all 42 hand
   points, each as x, y and depth, plus two flags that say whether each hand was detected.
   27 + 63 + 63 + 2 = 155 values per frame.
2. **Normalisation.** Every point is expressed relative to the shoulder midpoint and divided by
   the shoulder width. Where the signer stands and how far they are from the camera stop
   mattering.
3. **Clips.** A sign is a movement, so clips are resampled to exactly 32 frames. Training
   clips are also mirrored (swapping the left and right hand blocks), sped up or slowed down,
   shifted and jittered.
4. **Classifier.** A two-layer bidirectional GRU (about 0.5 M weights) reads the 32 x 155
   table and scores every sign. It trains in minutes and runs in about 3 ms through ONNX
   Runtime.
5. **Evaluation.** Two numbers are always reported: the official KArSL split (new recordings
   of the same signers) and a held-out signer (a person the model never saw). The second is
   the honest one and is expected to be lower.

## The hardware

Everything here is developed and benchmarked on a deliberately weak machine, because that is
the hardware a school or a clinic actually has:

| Part | Spec | Consequence |
|---|---|---|
| CPU | AMD Ryzen 5 3500U, 4 cores / 8 threads | all inference and landmark extraction run here |
| RAM | 7 GB | datasets are streamed from disk, never loaded whole |
| GPU | Radeon Vega 8 integrated, 1 GB, no CUDA | no local deep-learning training; DirectML is the only GPU path |
| Disk | 195 GB free on D: | enough for KArSL-100 (19 GB of video) |

Training of the small GRU runs on this CPU. Anything heavier is planned for free Colab or
Kaggle sessions with the same code.

## Setup

```
conda create -p ./.env python=3.11 -y          # any 3.10-3.12 interpreter works
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python scripts/download_models.py             # MediaPipe holistic model, 14 MB, free
```

## Use

| Step | Command |
|---|---|
| Record your own clips | `python -m ssl_recognition.record --label hello --signer meshal --clips 10` |
| Prepare KArSL (after downloading the public archives into `data/raw`) | `python scripts/prepare_karsl.py --signers 01 02 03 --ranges 0071-0170` |
| Extract landmarks from videos | `python scripts/extract_landmarks.py --videos data/videos --signer-from-name --workers 3` |
| Train, official split | `python -m ssl_recognition.train --config configs/karsl100_s01.yaml` |
| Train, held-out signer | `python -m ssl_recognition.train --config configs/baseline.yaml --test-signer s03` |
| Export to ONNX (with parity check) | `python scripts/export_onnx.py --run runs/karsl100_s01` |
| Live webcam demo | `python -m ssl_recognition.infer --run runs/karsl100_s01 --onnx` |
| Benchmark this machine | `python scripts/benchmark_cpu.py --run runs/karsl100_s01` |
| Tests and lint | `pytest -q && ruff check .` |

Data layout: `data/landmarks/<label>/<signer>_<split>_<id>.npz`, one small file per clip.
Nothing under `data/` or `runs/` is committed.

## Results

See [`reports/results.md`](reports/results.md) for the full log. Headline numbers are copied
here as they land.

| Date | Data | Split | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|
| 2026-10-12 | 6 signs, signer 01 | official | 0.88 | 1.00 | smoke run on partial data, validates the loop only |
| 2026-10-12 | KArSL-100, signer 01, 100 classes | official (800 test clips) | **0.990** | 0.998 | BiGRU, 566k params, trained on the laptop CPU in 37 min. Signer-dependent. |

## Edge benchmark

Measured on the laptop above, 640x480 input:

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| MediaPipe holistic, per frame | 71 ms | 84 ms | the bottleneck, about 15 FPS |
| Classifier, PyTorch | 25 ms | 68 ms | |
| Classifier, ONNX Runtime CPU | 3.3 ms | 12 ms | 2.1 MB model, max output difference vs PyTorch 4.8e-7 |

Planned: lower capture resolution, INT8 quantisation, DirectML on the integrated GPU.

## Roadmap

1. Baseline GRU on KArSL-100, signer 01, official split
2. Signer-independent evaluation across all three signers
3. All 502 KArSL signs
4. ONNX INT8 and DirectML benchmark chapter
5. Live demo with sign-boundary detection
6. Continuous Saudi Sign Language on Isharah pose data (CTC over landmark sequences)

## Data

- [KArSL](https://hamzah-luqman.github.io/KArSL/): 502 isolated Arabic signs, 3 signers,
  75,300 samples (KFUPM). Raw videos and RGB frames are public Google Drive folders.
  KArSL-100 is signs 0071 to 0170.
- [Isharah](https://snalyami.github.io/Isharah_CSLR/): the first large-scale continuous
  Saudi Sign Language dataset, 30,000+ smartphone videos by deaf signers with pose data
  (IAU and KFUPM, academic use only).
- Own recordings through `ssl_recognition.record`.

## Licence

MIT for the code. Dataset licences are separate and belong to their owners.
