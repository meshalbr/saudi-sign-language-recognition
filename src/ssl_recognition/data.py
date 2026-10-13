"""Dataset utilities: loading landmark clips, resampling, augmentation and splits.

On-disk layout (one .npz per clip):
    data/landmarks/<label>/<signer>_<clip_id>.npz
        x      float32 (T, FEATURE_DIM)
        label  str
        signer str
        fps    float
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .landmarks import (
    FEATURE_DIM,
    LEFT_FLAG,
    LEFT_SLICE,
    RIGHT_FLAG,
    RIGHT_SLICE,
    mirror_features,
)


@dataclass
class Clip:
    x: np.ndarray
    label: str
    signer: str
    path: Path


def save_clip(path: Path, x: np.ndarray, label: str, signer: str, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32), label=label, signer=signer, fps=fps)


def load_clip(path: Path) -> Clip:
    with np.load(path, allow_pickle=False) as z:
        return Clip(x=z["x"], label=str(z["label"]), signer=str(z["signer"]), path=path)


def load_dataset(root: Path) -> list[Clip]:
    root = Path(root)
    clips = [load_clip(p) for p in sorted(root.glob("*/*.npz"))]
    if not clips:
        raise FileNotFoundError(f"No .npz clips found under {root}")
    return clips


def resample(x: np.ndarray, seq_len: int) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (seq_len, D)."""
    t = len(x)
    if t == seq_len:
        return x.astype(np.float32)
    if t == 1:
        return np.repeat(x, seq_len, axis=0).astype(np.float32)
    src = np.linspace(0.0, t - 1, num=seq_len)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None].astype(np.float32)
    return (x[lo] * (1 - w) + x[hi] * w).astype(np.float32)


def augment(x: np.ndarray, rng: random.Random, hand_drop: float = 0.0) -> np.ndarray:
    """Random mirror, time-scale, scale, shift, jitter and optional hand-block dropout."""
    y = x
    if rng.random() < 0.5:
        y = mirror_features(y)
    if hand_drop > 0 and rng.random() < hand_drop:
        # Simulate a missed hand: zero one hand block and its presence flag for the clip.
        y = y.copy()
        blk, flag = (LEFT_SLICE, LEFT_FLAG) if rng.random() < 0.5 else (RIGHT_SLICE, RIGHT_FLAG)
        y[:, blk] = 0.0
        y[:, flag] = 0.0
    # Temporal crop / stretch: keep 70-100% of the clip, random start.
    t = len(y)
    keep = max(4, int(t * rng.uniform(0.7, 1.0)))
    start = rng.randint(0, t - keep)
    y = y[start : start + keep]
    # Spatial scale and shift on coordinate blocks (not on the two presence flags).
    coords = y[:, : FEATURE_DIM - 2]
    scale = rng.uniform(0.9, 1.1)
    shift = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0], dtype=np.float32)
    coords = coords.reshape(len(y), -1, 3) * scale + shift
    coords = (
        coords + rng.gauss(0, 1) * 0.0 + np.random.normal(0, 0.01, coords.shape).astype(np.float32)
    )
    y = np.concatenate([coords.reshape(len(y), -1), y[:, FEATURE_DIM - 2 :]], axis=1)
    return y.astype(np.float32)


def build_label_map(clips: list[Clip]) -> dict[str, int]:
    labels = sorted({c.label for c in clips})
    return {lab: i for i, lab in enumerate(labels)}


def split_clips(
    clips: list[Clip],
    seed: int = 42,
    val_frac: float = 0.15,
    test_signer: str | None = None,
    official: bool = False,
) -> tuple[list[Clip], list[Clip], list[Clip]]:
    """Split into train/val/test.

    ``test_signer``: every clip of that signer becomes the test set (signer-independent).
    ``official``: clips named ``*_test_*`` form the test set (KArSL's published split).
    Otherwise a stratified random 70/15/15 split. Validation is always carved from train.
    """
    rng = random.Random(seed)
    if test_signer is not None:
        test = [c for c in clips if c.signer == test_signer]
        rest = [c for c in clips if c.signer != test_signer]
        if not test:
            raise ValueError(f"No clips for signer {test_signer!r}")
    elif official:
        # KArSL ships an official split encoded in the file name: <signer>_train_* / <signer>_test_*
        is_test = [c.path is not None and "_test_" in c.path.name for c in clips]
        test = [c for c, t in zip(clips, is_test, strict=True) if t]
        rest = [c for c, t in zip(clips, is_test, strict=True) if not t]
        if not test:
            raise ValueError("official split requested but no '_test_' clips found")
    else:
        test, rest = [], list(clips)

    by_label: dict[str, list[Clip]] = {}
    for c in rest:
        by_label.setdefault(c.label, []).append(c)
    train, val = [], []
    test_frac = 0.0 if (test_signer is not None or official) else val_frac
    for _, items in sorted(by_label.items()):
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        n_test = max(1, int(round(n * test_frac))) if (n >= 3 and test_frac > 0) else 0
        val.extend(items[:n_val])
        test.extend(items[n_val : n_val + n_test])
        train.extend(items[n_val + n_test :])
    return train, val, test


def save_labels(path: Path, label_map: dict[str, int]) -> None:
    path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> list[str]:
    label_map = json.loads(Path(path).read_text(encoding="utf-8"))
    return [lab for lab, _ in sorted(label_map.items(), key=lambda kv: kv[1])]


class ClipDataset:
    """Minimal torch-compatible dataset over Clip objects."""

    def __init__(
        self,
        clips: list[Clip],
        label_map: dict[str, int],
        seq_len: int,
        train: bool = False,
        seed: int = 0,
        cfg: dict | None = None,
    ):
        self.clips = clips
        self.label_map = label_map
        self.seq_len = seq_len
        self.train = train
        self.rng = random.Random(seed)
        self.cfg = cfg or {}
        self.hand_drop = float(self.cfg.get("hand_drop", 0.0))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        import torch

        from .features import transform

        c = self.clips[i]
        x = augment(c.x, self.rng, self.hand_drop) if self.train else c.x
        x = transform(resample(x, self.seq_len), self.cfg)
        return torch.from_numpy(x), self.label_map[c.label]
