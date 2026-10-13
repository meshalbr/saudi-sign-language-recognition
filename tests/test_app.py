"""Smoke test for the browser app: labels endpoint and one frame through the WebSocket.

Skipped when no trained run or the MediaPipe model is available (e.g. on CI).
"""

import json
from pathlib import Path

import numpy as np
import pytest

RUN = Path("runs/final_tf_all_official")


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

    if Path("data/landmarks/0071").is_dir():
        ref = client.get("/api/ref/0071").json()
        assert ref["frames"] and len(ref["frames"][0]["pose"]) == 9
    if Path("data/videos/0071").is_dir():
        r = client.get("/api/clip/0071")
        assert r.status_code == 200 and r.headers["content-type"].startswith("video/mp4")
    assert client.get("/api/ref/9999").status_code == 404
