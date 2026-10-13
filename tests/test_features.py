import random

import numpy as np

from ssl_recognition.data import augment
from ssl_recognition.features import COORD_DIM, feature_dim, transform
from ssl_recognition.landmarks import FEATURE_DIM, LEFT_FLAG, LEFT_SLICE, RIGHT_FLAG, RIGHT_SLICE


def _clip(t=10, seed=0):
    x = np.random.RandomState(seed).rand(t, FEATURE_DIM).astype(np.float32)
    x[:, LEFT_FLAG] = 1.0
    x[:, RIGHT_FLAG] = 0.0
    x[:, RIGHT_SLICE] = 0.0
    return x


def test_identity_without_options():
    x = _clip()
    y = transform(x, {})
    assert y.shape == x.shape and np.allclose(x, y)
    assert feature_dim({}) == FEATURE_DIM


def test_velocity_appends_differences():
    x = _clip()
    cfg = {"features": {"velocity": True}}
    y = transform(x, cfg)
    assert y.shape == (len(x), feature_dim(cfg)) == (len(x), FEATURE_DIM + COORD_DIM)
    assert np.allclose(y[0, FEATURE_DIM:], 0)
    assert np.allclose(y[1:, FEATURE_DIM:], x[1:, :COORD_DIM] - x[:-1, :COORD_DIM])


def test_hand_local_is_scale_invariant_and_zero_when_absent():
    x = _clip()
    cfg = {"features": {"hand_local": True}}
    y1 = transform(x, cfg)
    x2 = x.copy()
    x2[:, LEFT_SLICE] = (x[:, LEFT_SLICE].reshape(len(x), 21, 3) * 3.0 + 0.7).reshape(len(x), -1)
    y2 = transform(x2, cfg)
    assert np.allclose(y1[:, LEFT_SLICE], y2[:, LEFT_SLICE], atol=1e-4)
    assert not y1[:, RIGHT_SLICE].any()  # absent hand stays zero
    assert np.allclose(y1[:, :27], x[:, :27])  # pose block untouched


def test_hand_drop_zeroes_one_hand():
    x = _clip()
    x[:, RIGHT_FLAG] = 1.0
    x[:, RIGHT_SLICE] = 0.5
    y = augment(x, random.Random(3), hand_drop=1.0)
    flags = y[:, [LEFT_FLAG, RIGHT_FLAG]]
    assert set(np.unique(flags)) <= {0.0, 1.0}
    assert (flags.sum(axis=1) == 1).all()  # exactly one hand left per frame
