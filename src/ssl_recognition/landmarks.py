"""MediaPipe Holistic landmark extraction and conversion to a fixed feature vector.

Feature layout per frame (FEATURE_DIM = 155):
    [0:27]    9 upper-body pose landmarks (x, y, z), body-normalised
    [27:90]   21 left-hand landmarks (x, y, z), body-normalised
    [90:153]  21 right-hand landmarks (x, y, z), body-normalised
    [153]     left-hand present flag (0/1)
    [154]     right-hand present flag (0/1)

Body normalisation: subtract the shoulder midpoint and divide by the shoulder width, so the
features are invariant to where the signer stands and how far they are from the camera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Pose landmark indices (MediaPipe Pose topology): nose, shoulders, elbows, wrists, hips.
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
# Index pairs inside POSE_IDX that swap under a horizontal mirror (left <-> right).
POSE_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]
N_POSE = len(POSE_IDX)
N_HAND = 21
POSE_DIM = N_POSE * 3
HAND_DIM = N_HAND * 3
FEATURE_DIM = POSE_DIM + 2 * HAND_DIM + 2

LEFT_SLICE = slice(POSE_DIM, POSE_DIM + HAND_DIM)
RIGHT_SLICE = slice(POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM)
LEFT_FLAG = FEATURE_DIM - 2
RIGHT_FLAG = FEATURE_DIM - 1

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "holistic_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)


def _to_array(landmarks, n: int) -> np.ndarray | None:
    """Convert a MediaPipe landmark list to an (n, 3) array, or None when absent."""
    if not landmarks:
        return None
    # Tasks API may return List[NormalizedLandmark] or List[List[NormalizedLandmark]].
    first = landmarks[0]
    if isinstance(first, list):
        if not first:
            return None
        landmarks = first
    if len(landmarks) < n:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks[:n]], dtype=np.float32)


def features_from_arrays(
    pose: np.ndarray | None,
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> np.ndarray:
    """Build the 155-d feature vector from raw (33,3) pose and (21,3) hand arrays.

    Any of the inputs may be None. With no pose, the frame is all zeros.
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    if pose is None:
        return feat
    sel = pose[POSE_IDX]
    l_sh, r_sh = pose[11], pose[12]
    center = (l_sh + r_sh) / 2.0
    scale = float(np.linalg.norm((l_sh - r_sh)[:2]))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    feat[:POSE_DIM] = ((sel - center) / scale).reshape(-1)
    if left is not None:
        feat[LEFT_SLICE] = ((left - center) / scale).reshape(-1)
        feat[LEFT_FLAG] = 1.0
    if right is not None:
        feat[RIGHT_SLICE] = ((right - center) / scale).reshape(-1)
        feat[RIGHT_FLAG] = 1.0
    return feat


def features_from_result(result) -> np.ndarray:
    """Build the feature vector from a HolisticLandmarkerResult."""
    pose = _to_array(result.pose_landmarks, 33)
    left = _to_array(result.left_hand_landmarks, N_HAND)
    right = _to_array(result.right_hand_landmarks, N_HAND)
    return features_from_arrays(pose, left, right)


def mirror_features(x: np.ndarray) -> np.ndarray:
    """Horizontally mirror a (T, FEATURE_DIM) sequence: negate x, swap left/right."""
    y = x.copy()
    pose = y[:, :POSE_DIM].reshape(len(y), N_POSE, 3)
    for a, b in POSE_MIRROR_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    pose[:, :, 0] *= -1
    y[:, :POSE_DIM] = pose.reshape(len(y), -1)

    left = y[:, LEFT_SLICE].reshape(len(y), N_HAND, 3).copy()
    right = y[:, RIGHT_SLICE].reshape(len(y), N_HAND, 3).copy()
    left[:, :, 0] *= -1
    right[:, :, 0] *= -1
    y[:, LEFT_SLICE] = right.reshape(len(y), -1)
    y[:, RIGHT_SLICE] = left.reshape(len(y), -1)
    y[:, [LEFT_FLAG, RIGHT_FLAG]] = y[:, [RIGHT_FLAG, LEFT_FLAG]]
    return y


def hands_present(x: np.ndarray) -> np.ndarray:
    """Boolean per frame: at least one hand detected."""
    return (x[:, LEFT_FLAG] + x[:, RIGHT_FLAG]) > 0


class HolisticExtractor:
    """Thin wrapper around MediaPipe HolisticLandmarker for video streams.

    Usage:
        with HolisticExtractor() as ex:
            feat = ex.process(bgr_frame, timestamp_ms)
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run: python scripts/download_models.py"
            )
        self._mp = mp
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_conf,
            min_pose_landmarks_confidence=min_conf,
            min_hand_landmarks_confidence=min_conf,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        self._last_ts = -1
        self.last_result = None

    def process(self, bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Run the landmarker on a BGR frame and return the 155-d feature vector."""
        import cv2

        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self.last_result = self._landmarker.detect_for_video(image, timestamp_ms)
        return features_from_result(self.last_result)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_result(bgr: np.ndarray, result) -> np.ndarray:
    """Draw pose and hand points on a BGR frame (simple dots, no MediaPipe drawing utils)."""
    import cv2

    h, w = bgr.shape[:2]
    for lms, colour in (
        (result.pose_landmarks, (0, 200, 255)),
        (result.left_hand_landmarks, (0, 255, 0)),
        (result.right_hand_landmarks, (255, 0, 200)),
    ):
        arr = _to_array(lms, 21 if colour != (0, 200, 255) else 33)
        if arr is None:
            continue
        for x, y, _ in arr:
            cv2.circle(bgr, (int(x * w), int(y * h)), 3, colour, -1)
    return bgr
