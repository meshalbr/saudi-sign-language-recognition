"""Optional feature transforms applied on top of the stored 155-d landmark rows.

Configured through the training config and stored with the checkpoint, so training,
evaluation, ONNX export and the live demo all apply exactly the same transform.

    features:
      hand_local: true    # hand points relative to the wrist, scaled by hand size
      velocity: true      # append frame-to-frame differences of the coordinate block
"""

from __future__ import annotations

import numpy as np

from .landmarks import FEATURE_DIM, LEFT_FLAG, LEFT_SLICE, N_HAND, RIGHT_FLAG, RIGHT_SLICE

COORD_DIM = FEATURE_DIM - 2  # everything except the two presence flags
WRIST, MIDDLE_MCP = 0, 9


def feature_options(cfg: dict) -> dict:
    f = cfg.get("features") or {}
    return {
        "hand_local": bool(f.get("hand_local", False)),
        "velocity": bool(f.get("velocity", False)),
    }


def feature_dim(cfg: dict) -> int:
    opts = feature_options(cfg)
    return FEATURE_DIM + (COORD_DIM if opts["velocity"] else 0)


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
    if opts["velocity"]:
        coords = y[:, :COORD_DIM]
        vel = np.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]
        y = np.concatenate([y, vel], axis=1)
    return y
