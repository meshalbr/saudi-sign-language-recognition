"""Smoke test for the browser app: labels endpoint and one frame through the WebSocket.

Skipped when no trained run or the MediaPipe model is available (e.g. on CI).
"""

import json
from pathlib import Path

import numpy as np
import pytest

RUN = Path("runs/final_v2")


@pytest.mark.skipif(
    not (RUN / "model.onnx").exists() or not Path("models/holistic_landmarker.task").exists(),
    reason="needs a trained run and the MediaPipe model on this machine",
)
def test_labels_and_one_frame():
    import cv2
    from fastapi.testclient import TestClient

    from app import server

    server.STATE.update(run=str(RUN), names="data/karsl_labels.json", window=8, stride=1)
    client = TestClient(server.app)
    labels = client.get("/api/labels").json()
    assert len(labels) == 100 and {"id", "ar", "en"} <= set(labels[0])

    ok, jpg = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
    assert ok
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(jpg.tobytes())
        msg = json.loads(ws.receive_text())
    assert {"top", "hands", "lm", "ms"} <= set(msg)
    assert msg["hands"] == 0 and msg["top"] == []

    with client.websocket_connect("/ws_feat") as ws:
        ws.send_text(json.dumps({"f": [0.0] * 155}))
        msg = json.loads(ws.receive_text())
    assert msg["hands"] == 0 and msg["top"] == [] and "ms" in msg
    assert client.get("/models/holistic_landmarker.task").status_code == 200

    # Final-answer path: a real test clip at 30 fps, then hands down -> exactly one "final".
    clips = sorted(Path("data/landmarks/0071").glob("s01_test_*.npz"))
    if clips:
        x = np.load(clips[0])["x"]
        finals, shown = [], []
        rows = list(x) + [np.zeros(155, np.float32)] * 30
        with client.websocket_connect("/ws_display") as disp:
            with client.websocket_connect("/ws_feat?role=camera") as ws:
                for k, row in enumerate(rows):
                    ws.send_text(json.dumps({"f": [float(v) for v in row], "t": k * 33.3}))
                    m = json.loads(ws.receive_text())
                    shown.append(json.loads(disp.receive_text()))  # relayed to the laptop
                    if "final" in m:
                        finals.append(m["final"])
        assert len(finals) == 1 and finals[0][0]["id"] == "0071"
        assert any("final" in d for d in shown)

    if Path("data/landmarks/0071").is_dir():
        ref = client.get("/api/ref/0071").json()
        assert ref["frames"] and len(ref["frames"][0]["pose"]) == 9
    if Path("data/videos/0071").is_dir():
        r = client.get("/api/clip/0071")
        assert r.status_code == 200 and r.headers["content-type"].startswith("video/mp4")
    assert client.get("/api/ref/9999").status_code == 404


def _frame(wrist_y: float | None, hand: bool = True, hand_x: float = 0.0) -> np.ndarray:
    f = np.zeros(155, np.float32)
    if wrist_y is not None:
        f[:27] = 0.1  # some pose present
        f[16] = f[19] = wrist_y
    if hand:
        f[-1] = 1.0
        f[90:153:3] = hand_x  # right-hand x coordinates: the hand centre moves with hand_x
    return f


def _moving(n: int, wrist_y: float, start: int = 0) -> list:
    """n signing frames whose hand keeps moving (a real sign is never perfectly still)."""
    return [_frame(wrist_y, hand_x=0.3 * np.sin((start + k) / 3.0)) for k in range(n)]


def test_segmenter_keeps_short_dropouts_inside_one_sign():
    from app.server import Segmenter

    gone, rest = _frame(None, hand=False), _frame(2.3)
    seg = Segmenter(gap_ms=700)
    # "skull": 20 frames, 4-frame drop-out between the two movements, 25 frames, then hands down
    stream = _moving(20, 0.5) + [gone] * 4 + _moving(25, 0.5, 20) + [rest] * 30
    finals = [o for o in (seg.push(f, k * 33.3) for k, f in enumerate(stream)) if o is not None]
    assert len(finals) == 1
    assert len(finals[0]) == 49  # both movements plus the short gap, trailing rest trimmed


def test_segmenter_ignores_the_raise_from_rest():
    from app.server import Segmenter

    seg = Segmenter(gap_ms=700)
    rest, rising = _frame(2.3), _frame(1.9)
    # hands visible in the lap, raised through the rest line, signing, lowered back to the lap
    stream = [rest] * 30 + [rising] * 6 + _moving(30, 0.4) + [rising] * 6 + [rest] * 40
    finals = [o for o in (seg.push(f, k * 33.3) for k, f in enumerate(stream)) if o is not None]
    assert len(finals) == 1
    assert len(finals[0]) == 30  # only the frames above the rest line


@pytest.mark.skipif(not (RUN / "model.onnx").exists(), reason="needs a trained run")
def test_reference_endpoints_reject_paths_outside_the_dataset():
    from fastapi.testclient import TestClient

    from app import server

    server.STATE.update(run=str(RUN), names="data/karsl_labels.json", window=8, stride=1)
    client = TestClient(server.app)
    for bad in ("..%5C..%5Capp", "C:%5CWindows", "..", "0071%5C..%5C0072"):
        for ep in ("ref", "clip", "sprite"):
            assert client.get(f"/api/{ep}/{bad}").status_code == 404, (ep, bad)


def test_segmenter_caps_a_sign_at_six_seconds():
    from app.server import Segmenter

    seg = Segmenter(max_ms=6000)
    frames = _moving(400, 0.5)
    outs = [seg.push(f, k * 33.3) for k, f in enumerate(frames)]  # 13 s of continuous signing
    lengths = [len(o) for o in outs if o is not None]
    assert len(lengths) == 2 and all(170 <= n <= 182 for n in lengths)


def test_holding_still_finishes_the_sign_without_lowering_hands():
    from app.server import Segmenter

    seg = Segmenter(still_ms=1000)
    # 1 s of movement, then the hands stay up and still: no lowering, no leaving the frame
    stream = _moving(30, 0.5) + [_frame(0.5, hand_x=0.0)] * 60
    outs = [seg.push(f, k * 33.3) for k, f in enumerate(stream)]
    finals = [(k, o) for k, o in enumerate(outs) if o is not None]
    assert len(finals) == 1
    k, sign = finals[0]
    assert seg.end_reason == "hands still"
    assert 55 <= k <= 65  # about 1 s after the hands stopped
    assert len(sign) <= 30 + 12  # the waiting time is not part of the sign


def test_a_short_pause_inside_a_sign_does_not_finish_it():
    from app.server import Segmenter

    seg = Segmenter(still_ms=1000)
    pause = [_frame(0.5, hand_x=0.1)] * 15  # 0.5 s hold in the middle of the sign
    stream = _moving(30, 0.5) + pause + _moving(30, 0.5, 30) + [_frame(2.3)] * 30
    finals = [o for o in (seg.push(f, k * 33.3) for k, f in enumerate(stream)) if o is not None]
    assert len(finals) == 1 and len(finals[0]) == 75
