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


def test_fill_gaps_interpolates_short_gaps_only():
    from ssl_recognition.features import fill_gaps

    x = np.zeros((20, FEATURE_DIM), np.float32)
    x[:, RIGHT_FLAG] = 1.0
    x[:, RIGHT_SLICE] = np.arange(20, dtype=np.float32)[:, None]
    x[5:8, RIGHT_FLAG] = 0.0  # 3-frame gap: filled
    x[5:8, RIGHT_SLICE] = 0.0
    x[10:20, RIGHT_FLAG] = 0.0  # gap to the end: not filled
    x[10:20, RIGHT_SLICE] = 0.0
    y = fill_gaps(x, max_gap=8)
    assert (y[5:8, RIGHT_FLAG] == 1).all()
    assert np.allclose(y[5:8, RIGHT_SLICE.start], [5, 6, 7])
    assert (y[10:, RIGHT_FLAG] == 0).all()
    assert not y[:, LEFT_FLAG].any()


def test_augment_keeps_absent_hands_exactly_zero():
    x = _clip(t=30)  # left hand present, right hand absent
    for seed in range(20):
        y = augment(x, random.Random(seed), hand_drop=0.5)
        for blk, flag in ((LEFT_SLICE, LEFT_FLAG), (RIGHT_SLICE, RIGHT_FLAG)):
            absent = y[:, flag] <= 0
            assert not y[absent][:, blk].any()


def test_trajectory_is_the_same_wherever_the_movement_is_made():
    from ssl_recognition.features import TRAJ_DIM, transform

    x = _clip(t=20)
    path = np.linspace(0, 0.5, 20, dtype=np.float32)
    blk = x[:, LEFT_SLICE].reshape(20, 21, 3)
    blk[:, :, 0] = 0.2 + path[:, None]  # hand moves to the right
    x[:, LEFT_SLICE] = blk.reshape(20, -1)
    moved = x.copy()
    b2 = moved[:, LEFT_SLICE].reshape(20, 21, 3)
    b2[:, :, :2] += np.array(
        [0.7, -0.4], np.float32
    )  # same movement, elsewhere in front of the body
    moved[:, LEFT_SLICE] = b2.reshape(20, -1)
    cfg = {"features": {"trajectory": True}}
    a, b = transform(x, cfg)[:, -TRAJ_DIM:], transform(moved, cfg)[:, -TRAJ_DIM:]
    assert np.allclose(a, b, atol=1e-5)


def test_motion_trim_drops_still_frames_at_both_ends():
    from ssl_recognition.features import motion_trim

    x = _clip(t=40)
    blk = x[:, LEFT_SLICE].reshape(40, 21, 3)
    blk[:] = 0.3
    blk[10:30, :, 0] += np.linspace(0, 0.6, 20, dtype=np.float32)[:, None]  # moves only in 10-29
    blk[30:, :, 0] = blk[29, :, 0]
    x[:, LEFT_SLICE] = blk.reshape(40, -1)
    y = motion_trim(x, thr=0.01, min_keep=0.3)
    assert 18 <= len(y) <= 24


def test_smooth_keeps_absent_hands_zero_and_reduces_jitter():
    from ssl_recognition.features import smooth

    x = _clip(t=30)
    jitter = np.random.RandomState(0).normal(0, 0.05, (30, 63)).astype(np.float32)
    x[:, LEFT_SLICE] = 0.5 + jitter
    y = smooth(x, 3)
    assert y[:, LEFT_SLICE].std(axis=0).mean() < x[:, LEFT_SLICE].std(axis=0).mean()
    assert not y[:, RIGHT_SLICE].any()


def test_face_block_measures_hand_position_relative_to_nose_and_chin():
    from ssl_recognition.features import FACE_DIM, feature_dim, transform
    from ssl_recognition.landmarks import N_FACE

    x = _clip(t=10)  # left hand present
    blk = x[:, LEFT_SLICE].reshape(10, 21, 3)
    blk[:] = 0.0
    blk[:, 0, :2] = [0.1, -0.5]  # wrist just under the chin
    x[:, LEFT_SLICE] = blk.reshape(10, -1)
    face = np.zeros((10, N_FACE, 3), np.float32)
    face[:, 0, :2] = [0.0, -1.3]  # nose
    face[:, 1, :2] = [0.0, -0.7]  # chin
    cfg = {"features": {"face": True}}
    y = transform(x, cfg, face.reshape(10, -1))
    assert y.shape == (10, feature_dim(cfg)) and FACE_DIM == N_FACE * 3 + 24
    rel = y[:, -FACE_DIM + N_FACE * 3 :]
    # left hand, wrist relative to the nose then to the chin
    assert np.allclose(rel[0, 0:2], [0.1, 0.8], atol=1e-5)
    assert np.allclose(rel[0, 6:8], [0.1, 0.2], atol=1e-5)
    assert not rel[:, 12:].any()  # right hand absent


def test_mirror_face_is_an_involution():
    from ssl_recognition.landmarks import N_FACE, mirror_face

    f = np.random.RandomState(5).rand(7, N_FACE, 3).astype(np.float32)
    assert np.allclose(mirror_face(mirror_face(f)), f)


def test_dataset_carries_face_points_through_augmentation(tmp_path):
    from ssl_recognition.data import ClipDataset, load_dataset, save_clip
    from ssl_recognition.features import feature_dim
    from ssl_recognition.landmarks import N_FACE

    x = _clip(t=30)
    face = np.random.RandomState(1).rand(30, N_FACE, 3).astype(np.float32)
    save_clip(tmp_path / "0071" / "s01_000.npz", x, "0071", "s01", 30.0, face=face)
    clips = load_dataset(tmp_path)
    assert clips[0].face is not None and clips[0].face.shape == (30, N_FACE, 3)
    cfg = {
        "features": {"face": True, "hand_local": True, "velocity": True, "trajectory": True},
        "fill_gaps": 8,
        "preprocess": {"smooth": 3, "motion_trim": 0.01},
        "hand_shift": 0.2,
    }
    ds = ClipDataset(clips, {"0071": 0}, 32, train=True, cfg=cfg)
    xt, label = ds[0]
    assert tuple(xt.shape) == (32, feature_dim(cfg)) and label == 0
