import random

import numpy as np

from ssl_recognition.data import (
    Clip,
    augment,
    build_label_map,
    load_dataset,
    resample,
    save_clip,
    split_clips,
)
from ssl_recognition.landmarks import FEATURE_DIM


def _clip(label, signer, t=20, seed=0):
    x = np.random.RandomState(seed).rand(t, FEATURE_DIM).astype(np.float32)
    x[:, -2:] = 1.0
    return Clip(x=x, label=label, signer=signer, path=None)


def test_resample_shapes():
    x = np.arange(10, dtype=np.float32)[:, None].repeat(FEATURE_DIM, 1)
    assert resample(x, 32).shape == (32, FEATURE_DIM)
    assert resample(x, 5).shape == (5, FEATURE_DIM)
    assert np.isclose(resample(x, 5)[-1, 0], 9.0)
    assert resample(x[:1], 8).shape == (8, FEATURE_DIM)


def test_augment_keeps_dim_and_flags():
    x = _clip("a", "s1").x
    y = augment(x, random.Random(0))
    assert y.shape[1] == FEATURE_DIM
    assert set(np.unique(y[:, -2:])) <= {0.0, 1.0}


def test_split_by_signer():
    clips = [
        _clip(lab, s, seed=i)
        for i, (lab, s) in enumerate(
            [(lab, s) for lab in ("a", "b") for s in ("s1", "s2", "s3") for _ in range(3)]
        )
    ]
    train, val, test = split_clips(clips, test_signer="s3")
    assert all(c.signer == "s3" for c in test) and len(test) == 6
    assert all(c.signer != "s3" for c in train + val)
    assert len(train) + len(val) == 12


def test_official_split_uses_test_token_in_filename(tmp_path):
    from pathlib import Path

    clips = []
    for lab in ("a", "b"):
        for i in range(6):
            c = _clip(lab, "s1", seed=i)
            c.path = Path(f"{lab}/s1_train_{i:03d}.npz")
            clips.append(c)
        for i in range(2):
            c = _clip(lab, "s1", seed=10 + i)
            c.path = Path(f"{lab}/s1_test_{i:03d}.npz")
            clips.append(c)
    train, val, test = split_clips(clips, official=True)
    assert len(test) == 4 and all("_test_" in c.path.name for c in test)
    assert len(train) + len(val) == 12 and all("_train_" in c.path.name for c in train + val)


def test_save_and_load_roundtrip(tmp_path):
    c = _clip("hello", "s1")
    save_clip(tmp_path / "hello" / "s1_000.npz", c.x, c.label, c.signer, 30.0)
    save_clip(tmp_path / "bye" / "s1_000.npz", c.x, "bye", c.signer, 30.0)
    clips = load_dataset(tmp_path)
    assert {c.label for c in clips} == {"hello", "bye"}
    assert build_label_map(clips) == {"bye": 0, "hello": 1}
