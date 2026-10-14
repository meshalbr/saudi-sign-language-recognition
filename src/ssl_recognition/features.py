"""Optional feature transforms applied on top of the stored 155-d landmark rows.

Configured through the training config and stored with the checkpoint, so training,
evaluation, ONNX export and the live demo all apply exactly the same transform.

    features:
      hand_local: true    # hand points relative to the wrist, scaled by hand size
      velocity: true      # append frame-to-frame differences of the coordinate block
      trajectory: true    # append each hand's path relative to its own average position
    preprocess:
      smooth: 3           # moving-average window (frames) against landmark jitter
      motion_trim: 0.01   # cut still frames at the start and end (hand-centre speed threshold)
"""

from __future__ import annotations

import numpy as np

from .landmarks import FEATURE_DIM, LEFT_FLAG, LEFT_SLICE, N_HAND, RIGHT_FLAG, RIGHT_SLICE

COORD_DIM = FEATURE_DIM - 2  # everything except the two presence flags
WRIST, MIDDLE_MCP = 0, 9


TRAJ_POINTS = (0, 4, 8, 12, 16, 20)  # wrist and the five fingertips
TRAJ_DIM = 2 * len(TRAJ_POINTS) * 2  # two hands, x and y


def feature_options(cfg: dict) -> dict:
    f = cfg.get("features") or {}
    return {
        "hand_local": bool(f.get("hand_local", False)),
        "velocity": bool(f.get("velocity", False)),
        "trajectory": bool(f.get("trajectory", False)),
    }


def feature_dim(cfg: dict) -> int:
    opts = feature_options(cfg)
    return (
        FEATURE_DIM
        + (COORD_DIM if opts["velocity"] else 0)
        + (TRAJ_DIM if opts["trajectory"] else 0)
    )


def _trajectory(x: np.ndarray) -> np.ndarray:
    """(T, 155) -> (T, 24): wrist and fingertip paths of each hand relative to that hand's mean
    wrist position over the clip. The same movement gives the same numbers wherever in front of
    the body it is performed; absent frames are zero."""
    out = np.zeros((len(x), TRAJ_DIM), dtype=np.float32)
    for k, (blk, flag) in enumerate(((LEFT_SLICE, LEFT_FLAG), (RIGHT_SLICE, RIGHT_FLAG))):
        present = x[:, flag] > 0
        if not present.any():
            continue
        h = x[:, blk].reshape(len(x), N_HAND, 3)[:, list(TRAJ_POINTS), :2]
        centre = h[present, 0].mean(axis=0)
        rel = h - centre
        rel[~present] = 0.0
        out[:, k * 12 : (k + 1) * 12] = rel.reshape(len(x), -1)
    return out


def _hand_local(block: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """(T, 63) hand block -> wrist-relative, hand-size-normalised; zeros where absent."""
    h = block.reshape(len(block), N_HAND, 3)
    wrist = h[:, WRIST : WRIST + 1]
    size = np.linalg.norm((h[:, MIDDLE_MCP] - h[:, WRIST])[:, :2], axis=1)
    size = np.where(size < 1e-4, 1.0, size)[:, None, None]
    out = (h - wrist) / size
    out[flag <= 0] = 0.0
    return out.reshape(len(block), -1).astype(np.float32)


def transform(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply the configured transforms to a (T, 155) clip. Returns (T, feature_dim(cfg))."""
    opts = feature_options(cfg)
    y = x.astype(np.float32, copy=True)
    if opts["hand_local"]:
        y[:, LEFT_SLICE] = _hand_local(y[:, LEFT_SLICE], y[:, LEFT_FLAG])
        y[:, RIGHT_SLICE] = _hand_local(y[:, RIGHT_SLICE], y[:, RIGHT_FLAG])
    traj = _trajectory(x) if opts["trajectory"] else None  # from the body-frame hands
    if opts["velocity"]:
        coords = y[:, :COORD_DIM]
        vel = np.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]
        y = np.concatenate([y, vel], axis=1)
    if traj is not None:
        y = np.concatenate([y, traj], axis=1)
    return y


def fill_gaps(x: np.ndarray, max_gap: int = 8) -> np.ndarray:
    """Linearly interpolate a hand that MediaPipe lost for at most ``max_gap`` frames.

    Only gaps with the same hand detected on both sides are filled; longer gaps and missing hands
    at the start or end stay missing. The pose block is untouched.
    """
    y = x.astype(np.float32, copy=True)
    for blk, flag in ((LEFT_SLICE, LEFT_FLAG), (RIGHT_SLICE, RIGHT_FLAG)):
        present = y[:, flag] > 0
        idx = np.where(present)[0]
        for a, b in zip(idx[:-1], idx[1:], strict=False):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                w = (np.arange(1, gap + 1) / (gap + 1))[:, None].astype(np.float32)
                y[a + 1 : b, blk] = y[a, blk] * (1 - w) + y[b, blk] * w
                y[a + 1 : b, flag] = 1.0
    return y


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Centred moving average over time against landmark jitter. Each block is averaged only over
    frames where it is present, so a missing hand never pulls a present one towards zero."""
    if k <= 1 or len(x) < 3:
        return x
    y = x.astype(np.float32, copy=True)
    half = k // 2
    for blk, flag in ((slice(0, 27), None), (LEFT_SLICE, LEFT_FLAG), (RIGHT_SLICE, RIGHT_FLAG)):
        present = (np.abs(x[:, blk]).sum(axis=1) > 0) if flag is None else (x[:, flag] > 0)
        vals = x[:, blk]
        for t in np.where(present)[0]:
            lo, hi = max(0, t - half), min(len(x), t + half + 1)
            y[t, blk] = vals[lo:hi][present[lo:hi]].mean(axis=0)
    return y


def hand_speed(x: np.ndarray) -> np.ndarray:
    """Per-frame speed of the faster hand centre, in shoulder widths per frame."""
    sp = np.zeros(len(x), dtype=np.float32)
    for blk, flag in ((LEFT_SLICE, LEFT_FLAG), (RIGHT_SLICE, RIGHT_FLAG)):
        c = x[:, blk].reshape(len(x), N_HAND, 3)[:, :, :2].mean(axis=1)
        present = x[:, flag] > 0
        both = present[1:] & present[:-1]
        d = np.zeros(len(x), dtype=np.float32)
        d[1:][both] = np.linalg.norm(c[1:] - c[:-1], axis=1)[both]
        sp = np.maximum(sp, d)
    return sp


def motion_trim(x: np.ndarray, thr: float = 0.01, min_keep: float = 0.6) -> np.ndarray:
    """Drop still frames at the start and end so that holds before and after the movement do not
    count. Keeps at least ``min_keep`` of the clip, centred on the movement."""
    n = len(x)
    if n < 8 or thr <= 0:
        return x
    moving = np.where(hand_speed(x) > thr)[0]
    if len(moving) == 0:
        return x
    a, b = max(0, int(moving[0]) - 1), min(n, int(moving[-1]) + 2)
    need = int(np.ceil(n * min_keep))
    if b - a < need:
        mid = (a + b) // 2
        a = max(0, min(mid - need // 2, n - need))
        b = a + need
    return x[a:b]


def preprocess(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Clip-level cleaning shared by training and inference: gap filling, smoothing, trimming."""
    pp = cfg.get("preprocess") or {}
    if cfg.get("fill_gaps"):
        x = fill_gaps(x, int(cfg["fill_gaps"]))
    if pp.get("smooth"):
        x = smooth(x, int(pp["smooth"]))
    if pp.get("motion_trim"):
        x = motion_trim(x, float(pp["motion_trim"]))
    return x
