"""The in-browser engine (app/static/engine.js) must compute exactly what the Python code does.

Runs the JavaScript under Node on the same clips and compares:
- the model input (fill gaps, smoothing, motion trim, resampling, hand-local, velocity,
  trajectory and face features) with Predictor.predict's preprocessing,
- where the Segmenter cuts a live stream into signs, and why,
- the start/end grid best_window searches.
Clips are synthetic (seeded), so the test needs no dataset and runs on CI.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.server import Segmenter
from ssl_recognition.data import resample
from ssl_recognition.features import preprocess, transform

ROOT = Path(__file__).resolve().parents[1]
CFG = {
    "seq_len": 32,
    "fill_gaps": 8,
    "features": {"hand_local": True, "velocity": True, "trajectory": True, "face": True},
    "preprocess": {"smooth": 3, "motion_trim": 0.01},
}

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


def synthetic_clip(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A plausible landmark clip: body near the origin, hands moving above the rest line,
    with short and long drop-outs of each hand and some frames without a face."""
    x = np.zeros((n, 155), np.float32)
    x[:, :27] = rng.normal(0, 0.3, (n, 27))
    x[:, 16] = rng.uniform(0.2, 1.2, n)  # wrists raised
    x[:, 19] = rng.uniform(0.2, 1.2, n)
    for a0, flag in ((27, 153), (90, 154)):
        base = rng.normal(0, 0.5, (1, 63))
        walk = np.cumsum(rng.normal(0, 0.03, (n, 63)), axis=0)
        x[:, a0 : a0 + 63] = base + walk
        present = rng.random(n) > 0.15
        if n > 30:
            present[10:13] = False  # short gap, gets filled
            present[20 : 20 + int(rng.integers(9, 12))] = False  # long gap, stays missing
        x[~present, a0 : a0 + 63] = 0
        x[:, flag] = present.astype(np.float32)
    face = rng.normal(0, 0.4, (n, 13, 3)).astype(np.float32)
    face[rng.random(n) < 0.1] = 0
    return x, face


def python_model_input(x: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Predictor.predict without the network."""
    xp, fp = preprocess(x, CFG, face)
    f_rs = resample(fp.reshape(len(fp), -1), CFG["seq_len"])
    return transform(resample(xp, CFG["seq_len"]), CFG, f_rs)


def stream(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rest, sign, hold still, rest, sign, lower hands: exercises every way a sign ends."""
    parts = []
    for kind, n in (("rest", 15), ("sign", 40), ("hold", 40), ("rest", 30), ("sign", 35),
                    ("rest", 30), ("sign", 200), ("rest", 30), ("sign", 25)):
        x, _ = synthetic_clip(rng, n)
        x[:, 153:] = 1
        if kind == "rest":
            x[:, 16] = x[:, 19] = 2.2
        if kind == "hold":
            x[:, 27:153] = x[0, 27:153]
        parts.append(x)
    x = np.concatenate(parts)
    t = np.arange(len(x), dtype=np.float64) * (1000 / 30)
    face = rng.normal(0, 0.4, (len(x), 39)).astype(np.float32)
    return x, t, face


def python_segments(x, t, face) -> list:
    seg = Segmenter()
    events = []
    for i in range(len(x)):
        done = seg.push(x[i], float(t[i]), face[i].reshape(13, 3))
        if done is not None:
            events.append([i, len(done), seg.end_reason])
    tail = seg.flush()
    if tail is not None:
        events.append([len(x), len(tail), seg.end_reason])
    return events


def test_engine_matches_python(tmp_path):
    rng = np.random.default_rng(7)
    clips = [synthetic_clip(rng, n) for n in (5, 16, 31, 32, 45, 70, 140)]
    streams = [stream(rng) for _ in range(3)]
    grid_sizes = list(range(16, 200))
    cases = {
        "cfg": CFG,
        "clips": [{"x": x.tolist(), "face": f.reshape(len(f), -1).tolist()} for x, f in clips],
        "streams": [{"x": x.tolist(), "t": t.tolist(), "face": f.tolist()} for x, t, f in streams],
        "grid_sizes": grid_sizes,
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    run = subprocess.run(
        ["node", str(ROOT / "tests" / "engine_parity.js"), str(path)],
        capture_output=True, text=True, check=True,
    )
    js = json.loads(run.stdout)

    for (x, f), got in zip(clips, js["inputs"], strict=True):
        want = python_model_input(x, f)
        got = np.asarray(got, np.float32)
        assert got.shape == want.shape == (32, 395)
        np.testing.assert_allclose(got, want, atol=2e-4, rtol=1e-4)

    for (x, t, f), got in zip(streams, js["segments"], strict=True):
        assert got == python_segments(x, t, f)

    for n, (starts, ends) in zip(grid_sizes, js["grids"], strict=True):
        assert starts == sorted({int(v) for v in np.linspace(0, int(n * 0.45), 10)})
        assert ends == sorted({int(v) for v in np.linspace(int(n * 0.65), n, 7)})
