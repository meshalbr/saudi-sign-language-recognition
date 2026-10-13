"""Simulate the real-world problem: the signer raises a hand from the lap to the start position
before the sign, and lowers it afterwards. Adds realistic approach/retraction frames to real
KArSL test clips (the active hands and wrists are translated, not just one coordinate) and
reports how often the live pipeline still gives exactly one correct answer."""

from __future__ import annotations

import glob
import random

import numpy as np

REST = 2.4  # wrist height in the lap, shoulder widths below the shoulder line
L_HAND, R_HAND = slice(27, 90), slice(90, 153)
L_WRIST_Y, R_WRIST_Y = 16, 19  # pose wrists: index 5 -> left, 6 -> right (feature order)


def shifted(frame: np.ndarray, dy: float) -> np.ndarray:
    """Move every present hand (and its pose wrist/elbow) down by dy."""
    g = frame.copy()
    for hand, flag, wy, ey in ((L_HAND, 153, 16, 10), (R_HAND, 154, 19, 13)):
        if g[flag] > 0:
            blk = g[hand].reshape(21, 3)
            blk[:, 1] += dy
            g[hand] = blk.reshape(-1)
            g[wy] += dy
            g[ey] += dy * 0.5
    return g


def with_raise(x: np.ndarray, n_up: int = 15, n_down: int = 15, hold: int = 25) -> np.ndarray:
    first, last = x[0], x[-1]
    y0 = min(first[L_WRIST_Y], first[R_WRIST_Y])
    y1 = min(last[L_WRIST_Y], last[R_WRIST_Y])
    up = [shifted(first, d) for d in np.linspace(REST - y0, 0, n_up, endpoint=False)]
    down = [shifted(last, d) for d in np.linspace(0, REST - y1, n_down + 1)[1:]]
    rest = [shifted(last, REST - y1)] * hold
    return np.stack(up + list(x) + down + rest)


def run(n: int = 200, seed: int = 1, run_dir: str = "runs/final_tf_all_official") -> None:
    from app import server

    server.STATE.update(run=run_dir, names="data/karsl_labels.json", window=40, stride=1)
    server.STATE.pop("predictor", None)
    rng = random.Random(seed)
    files = rng.sample(sorted(glob.glob("data/landmarks/*/*_test_*.npz")), n)
    stats = {"clean": [0, 0], "raised": [0, 0]}
    for f in files:
        x = np.load(f)["x"]
        y = f.replace("\\", "/").split("/")[-2]
        for name, seq in (("clean", np.concatenate([x, [shifted(x[-1], 3)] * 25])),
                          ("raised", with_raise(x))):
            s = server.Session()
            fin = []
            for k, row in enumerate(seq):
                o = s.step(row, k * 33.3)
                if "final" in o:
                    fin.append(o["final"][0]["id"])
            stats[name][0] += len(fin) == 1
            stats[name][1] += fin == [y]
    for name, (one, ok) in stats.items():
        print(f"{name:7s}: one answer {one}/{n}, correct {ok}/{n} ({ok / n:.1%})")


if __name__ == "__main__":
    run()
