"""Measure end-to-end latency on this machine: landmark extraction and classifier.

    python scripts/benchmark_cpu.py --run runs/baseline --video data/sample.mp4
    python scripts/benchmark_cpu.py --run runs/baseline            # synthetic frames only

Reports p50 / p95 latency and FPS for: MediaPipe holistic, classifier (torch), classifier (onnx).
This is the seed of the "edge benchmark" chapter (CPU vs DirectML, INT8 vs FP32).
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from ssl_recognition.landmarks import FEATURE_DIM, HolisticExtractor


def stats(ms: list[float]) -> str:
    a = np.array(ms)
    p50, p95, fps = np.percentile(a, 50), np.percentile(a, 95), 1000 / a.mean()
    return f"p50 {p50:6.1f} ms  p95 {p95:6.1f} ms  {fps:5.1f} FPS"


def bench_landmarker(frames: list[np.ndarray]) -> list[float]:
    times = []
    with HolisticExtractor() as ex:
        for i, f in enumerate(frames):
            t = time.perf_counter()
            ex.process(f, i * 33)
            times.append((time.perf_counter() - t) * 1000)
    return times[5:]  # drop warm-up


def bench_classifier(run_dir: Path, n: int, use_onnx: bool) -> list[float]:
    from ssl_recognition.infer import Predictor

    p = Predictor(run_dir, use_onnx=use_onnx)
    x = np.random.randn(40, FEATURE_DIM).astype(np.float32)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        p.predict(x)
        times.append((time.perf_counter() - t) * 1000)
    return times[5:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    print(f"machine: {platform.processor()} | {platform.system()} {platform.release()}")
    if args.video:
        import cv2

        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    else:
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(args.frames)
        ]
    print(f"holistic landmarker ({len(frames)} frames, 640x480): {stats(bench_landmarker(frames))}")
    if args.run:
        run = Path(args.run)
        print(f"classifier torch:  {stats(bench_classifier(run, 200, False))}")
        if (run / "model.onnx").exists():
            print(f"classifier onnx:   {stats(bench_classifier(run, 200, True))}")


if __name__ == "__main__":
    main()
