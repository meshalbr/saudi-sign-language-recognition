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


def test_features_are_identical_across_camera_aspect_ratios():
    """The same physical pose filmed by a 16:9, a 4:3 and an upright 9:16 camera must give the
    same features once each frame's aspect ratio is passed in."""
    rng = np.random.RandomState(4)
    # physical coordinates in metres, centred in the frame; z in the same units as x
    pose_m = rng.uniform(-0.4, 0.4, (33, 3)).astype(np.float32)
    pose_m[11] = [-0.2, -0.1, 0.0]
    pose_m[12] = [0.2, -0.1, 0.0]
    hand_m = rng.uniform(-0.3, 0.3, (21, 3)).astype(np.float32)

    def project(pts, w_m, h_m):  # MediaPipe-style: x / frame width, y / frame height, z / width
        out = pts.copy()
        out[:, 0] = pts[:, 0] / w_m + 0.5
        out[:, 1] = pts[:, 1] / h_m + 0.5
        out[:, 2] = pts[:, 2] / w_m
        return out

    feats = []
    for w_m, h_m in ((1.6, 0.9), (1.2, 0.9), (0.9, 1.6)):
        feats.append(
            features_from_arrays(
                project(pose_m, w_m, h_m), project(hand_m, w_m, h_m), None, aspect=w_m / h_m
            )
        )
    assert np.allclose(feats[0], feats[1], atol=1e-5)
    assert np.allclose(feats[0], feats[2], atol=1e-5)
