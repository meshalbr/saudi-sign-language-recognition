import numpy as np

from ssl_recognition.landmarks import (
    FEATURE_DIM,
    LEFT_FLAG,
    LEFT_SLICE,
    RIGHT_FLAG,
    RIGHT_SLICE,
    features_from_arrays,
    hands_present,
    mirror_features,
)


def _pose():
    rng = np.random.RandomState(0)
    pose = rng.rand(33, 3).astype(np.float32)
    pose[11] = [0.4, 0.5, 0.0]  # left shoulder
    pose[12] = [0.6, 0.5, 0.0]  # right shoulder
    return pose


def test_feature_dim_and_flags():
    pose = _pose()
    hand = np.random.RandomState(1).rand(21, 3).astype(np.float32)
    f = features_from_arrays(pose, hand, None)
    assert f.shape == (FEATURE_DIM,)
    assert f[LEFT_FLAG] == 1.0 and f[RIGHT_FLAG] == 0.0
    assert np.all(f[RIGHT_SLICE] == 0)
    assert np.any(f[LEFT_SLICE] != 0)


def test_no_pose_gives_zeros():
    f = features_from_arrays(None, None, None)
    assert f.shape == (FEATURE_DIM,) and not f.any()


def test_normalisation_is_translation_and_scale_invariant():
    pose = _pose()
    hand = np.random.RandomState(2).rand(21, 3).astype(np.float32)
    f1 = features_from_arrays(pose, hand, hand)
    pose2 = pose * 2.0 + np.array([0.3, -0.2, 0.0], dtype=np.float32)
    hand2 = hand * 2.0 + np.array([0.3, -0.2, 0.0], dtype=np.float32)
    f2 = features_from_arrays(pose2, hand2, hand2)
    assert np.allclose(f1, f2, atol=1e-5)


def test_mirror_is_an_involution_and_swaps_hands():
    pose = _pose()
    left = np.random.RandomState(3).rand(21, 3).astype(np.float32)
    f = features_from_arrays(pose, left, None)[None]
    m = mirror_features(f)
    assert m[0, RIGHT_FLAG] == 1.0 and m[0, LEFT_FLAG] == 0.0
    assert np.allclose(mirror_features(m), f)
    assert hands_present(f).tolist() == [True]
