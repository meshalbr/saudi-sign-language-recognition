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

# All KArSL videos are 1920x1080. MediaPipe returns x as a fraction of image width and y as a
# fraction of height, so features depend on the frame's aspect ratio. Live frames are converted
# to this aspect (x and z scaled by aspect / TRAIN_ASPECT) so they match training exactly.
TRAIN_ASPECT = 1920 / 1080

# Face reference points (MediaPipe face mesh indices). Many signs are defined by where the hand
# is relative to the face (nose vs mouth vs chin/throat); published KArSL work reports about +4%
# from adding face keypoints. Order: nose tip, chin, forehead, upper lip, lower lip, then mirror
# pairs (left, right): mouth corners, outer eye corners, inner eye corners, cheeks.
FACE_IDX = [1, 152, 10, 13, 14, 61, 291, 33, 263, 133, 362, 234, 454]
FACE_MIRROR_PAIRS = [(5, 6), (7, 8), (9, 10), (11, 12)]
N_FACE = len(FACE_IDX)

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "holistic_landmarker.task"
HAND_MODEL = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# Hand recovery: Holistic loses a hand lying flat on the chest or near the face (e.g. "trachea").
# When that happens and the pose still has the wrist, the standalone hand model is run on a crop
# around the wrist. Pose indices: left wrist 15, pinky 17, index 19; right 16, 18, 20.
RECOVER_SIDES = (("left", (15, 17, 19)), ("right", (16, 18, 20)))
RECOVER_MAX_WRIST_Y = 0.85  # only hands raised above this image height (resting hands are lower)
RECOVER_CROP = 0.55  # crop half-size in shoulder widths
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
    aspect: float = TRAIN_ASPECT,
) -> np.ndarray:
    """Build the 155-d feature vector from raw (33,3) pose and (21,3) hand arrays.

    ``aspect`` is the source frame's width / height. Any of the inputs may be None. With no
    pose, the frame is all zeros.
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    if pose is None:
        return feat
    k = float(aspect) / TRAIN_ASPECT
    if abs(k - 1.0) > 1e-6:
        fix = np.array([k, 1.0, k], dtype=np.float32)
        pose = pose * fix
        left = None if left is None else left * fix
        right = None if right is None else right * fix
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


def face_from_arrays(
    pose: np.ndarray | None, face: np.ndarray | None, aspect: float = TRAIN_ASPECT
) -> np.ndarray:
    """(N_FACE, 3) face reference points in the same body frame as the hand features
    (shoulder midpoint at the origin, one shoulder width = 1). Zeros if pose or face is missing."""
    out = np.zeros((N_FACE, 3), dtype=np.float32)
    if pose is None or face is None or len(face) <= max(FACE_IDX):
        return out
    k = float(aspect) / TRAIN_ASPECT
    fix = np.array([k, 1.0, k], dtype=np.float32)
    pose = pose * fix
    pts = face[FACE_IDX] * fix
    l_sh, r_sh = pose[11], pose[12]
    center = (l_sh + r_sh) / 2.0
    scale = float(np.linalg.norm((l_sh - r_sh)[:2]))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    out[:] = (pts - center) / scale
    return out


def mirror_face(f: np.ndarray) -> np.ndarray:
    """Mirror a (T, N_FACE, 3) face block: negate x and swap left/right points."""
    g = f.copy()
    for a, b in FACE_MIRROR_PAIRS:
        g[:, [a, b]] = g[:, [b, a]]
    g[:, :, 0] *= -1
    return g


def features_from_result(result, aspect: float = TRAIN_ASPECT) -> np.ndarray:
    """Build the feature vector from a HolisticLandmarkerResult (``aspect`` = frame W / H)."""
    pose = _to_array(result.pose_landmarks, 33)
    left = _to_array(result.left_hand_landmarks, N_HAND)
    right = _to_array(result.right_hand_landmarks, N_HAND)
    return features_from_arrays(pose, left, right, aspect)


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

    def __init__(
        self, model_path: str | Path = DEFAULT_MODEL, min_conf: float = 0.5, recover: bool = True
    ):
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
        self.recovered = {"left": None, "right": None}  # (21,3) arrays recovered this frame
        self.last_face = np.zeros((N_FACE, 3), dtype=np.float32)  # face block of the last frame
        self._hand = None
        if recover and Path(HAND_MODEL).exists():
            self._hand = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                )
            )

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
        h, w = bgr.shape[:2]
        pose = _to_array(self.last_result.pose_landmarks, 33)
        left = _to_array(self.last_result.left_hand_landmarks, N_HAND)
        right = _to_array(self.last_result.right_hand_landmarks, N_HAND)
        face = None
        if self.last_result.face_landmarks:
            f0 = self.last_result.face_landmarks
            f0 = f0[0] if isinstance(f0[0], list) else f0
            face = np.array([[lm.x, lm.y, lm.z] for lm in f0], dtype=np.float32)
        self.last_face = face_from_arrays(pose, face, aspect=w / h)
        self.recovered = {"left": None, "right": None}
        if self._hand is not None and pose is not None and (left is None or right is None):
            rec = recover_hands(self._hand, self._mp, rgb, pose, left is None, right is None)
            self.recovered = rec
            left = left if left is not None else rec["left"]
            right = right if right is not None else rec["right"]
        return features_from_arrays(pose, left, right, aspect=w / h)

    def close(self) -> None:
        self._landmarker.close()
        if self._hand is not None:
            self._hand.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def recover_hands(hand_model, mp, rgb, pose, need_left: bool, need_right: bool) -> dict:
    """Find missing hands with the standalone hand model on a crop around each raised wrist.

    Returns {"left": (21,3) or None, "right": (21,3) or None} in full-image normalised
    coordinates. A detected hand is accepted only if its wrist lies close to that side's pose
    wrist, so the other hand is never copied into the missing side.
    """
    h, w = rgb.shape[:2]
    out = {"left": None, "right": None}
    sw = float(np.linalg.norm((pose[11, :2] - pose[12, :2]) * [w, h]))
    if sw < 5:
        return out
    for side, (wr, pk, ix) in RECOVER_SIDES:
        if not ((side == "left" and need_left) or (side == "right" and need_right)):
            continue
        if pose[wr, 1] > RECOVER_MAX_WRIST_Y:
            continue
        cx = float(pose[[wr, pk, ix], 0].mean() * w)
        cy = float(pose[[wr, pk, ix], 1].mean() * h)
        half = max(16, int(sw * RECOVER_CROP))
        x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
        x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        res = hand_model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        best, best_d = None, None
        for lms in res.hand_landmarks or []:
            a = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            a[:, 0] = (x0 + a[:, 0] * (x1 - x0)) / w
            a[:, 1] = (y0 + a[:, 1] * (y1 - y0)) / h
            a[:, 2] = a[:, 2] * (x1 - x0) / w  # z is in crop-width units
            d = float(np.linalg.norm((a[0, :2] - pose[wr, :2]) * [w, h])) / sw
            if d < 0.6 and (best_d is None or d < best_d):
                best, best_d = a, d
        out[side] = best
    return out


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
