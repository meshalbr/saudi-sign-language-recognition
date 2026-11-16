"""End-to-end check of a running app (local, Docker or the public demo).

    python scripts/smoke_test.py --url http://localhost:7860

Checks the page and the API, that public mode blocks what it should, and then streams the
landmarks of real reference recordings through the WebSocket exactly like the browser does,
expecting the server to recognise the signs. Uses the reference pack from the v1.0 release.
Exits non-zero on any failure, so CI can use it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

REFS_URL = (
    "https://github.com/meshalbr/saudi-sign-language-recognition/releases/download/"
    "v1.0/references.zip"
)
SIGNS = ["0088", "0092", "0117", "0160", "0162", "0071", "0100", "0132"]


def get(url: str, method: str = "GET") -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_ready(base: str, seconds: int) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            if get(base + "/api/labels")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit(f"app not ready after {seconds} s")


def recognise(base: str, x: np.ndarray, face: np.ndarray | None) -> str | None:
    from websockets.sync.client import connect

    ws_url = base.replace("http", "ws", 1) + "/ws_feat"
    final = None
    with connect(ws_url, open_timeout=20) as ws:
        t = 0.0
        for i, row in enumerate(x):
            msg = {"f": row.tolist(), "t": t}
            if face is not None:
                msg["fa"] = face[i].reshape(-1).tolist()
            ws.send(json.dumps(msg))
            out = json.loads(ws.recv(timeout=20))
            final = out.get("final", final)
            t += 1000 / 30
        ws.send(json.dumps({"f": [0.0] * 155, "t": t, "end": True}))
        out = json.loads(ws.recv(timeout=20))
        final = out.get("final", final)
    return final[0]["id"] if final else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for startup")
    ap.add_argument("--min-correct", type=int, default=6, help=f"out of {len(SIGNS)} signs")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    fails: list[str] = []

    wait_ready(base, args.wait)
    status, body = get(base + "/")
    if status != 200 or b"Saudi Sign Language" not in body:
        fails.append(f"GET / -> {status}")
    labels = json.loads(get(base + "/api/labels")[1])
    if len(labels) != 100:
        fails.append(f"/api/labels has {len(labels)} signs, expected 100")
    info = json.loads(get(base + "/api/info")[1])
    print(f"info: {info}")
    if info.get("public"):
        for path, method in [("/api/save/0071", "POST"), ("/api/personalise", "POST"),
                             ("/api/model/personal", "POST")]:
            code = get(base + path, method)[0]
            if code != 403:
                fails.append(f"public mode: {method} {path} -> {code}, expected 403")

    with urllib.request.urlopen(REFS_URL, timeout=60) as r:
        refs = zipfile.ZipFile(io.BytesIO(r.read()))
    files = {n.split("/")[2]: n for n in refs.namelist() if n.endswith(".npz")}
    correct = 0
    for sid in SIGNS:
        with np.load(io.BytesIO(refs.read(files[sid]))) as z:
            x = z["x"].astype(np.float32)
            face = z["face"].astype(np.float32) if "face" in z.files else None
        t0 = time.perf_counter()
        got = recognise(base, x, face)
        ok = got == sid
        correct += ok
        print(f"{sid}: got {got}  {'ok' if ok else 'WRONG'}  ({len(x)} frames, "
              f"{(time.perf_counter() - t0) * 1000:.0f} ms)")
    if correct < args.min_correct:
        fails.append(f"recognised {correct}/{len(SIGNS)} signs, expected >= {args.min_correct}")

    if fails:
        print("FAILED:\n  " + "\n  ".join(fails))
        sys.exit(1)
    print(f"all checks passed, {correct}/{len(SIGNS)} signs recognised")


if __name__ == "__main__":
    main()
