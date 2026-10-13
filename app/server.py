"""Browser interface for live sign recognition: works on the laptop and on a phone.

    python -m app.server --run runs/final_tf_all_official            # http://localhost:8000
    python -m app.server --run runs/final_tf_all_official --https    # https://<ip>:8443, phones

The browser captures the camera, sends small JPEG frames over a WebSocket, and the laptop runs
MediaPipe + the classifier and streams back predictions and landmarks. Phones need HTTPS to
allow camera access, so --https creates a self-signed certificate in app/certs/ (ignored by git);
accept the browser warning once.
"""

from __future__ import annotations

import argparse
import collections
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ssl_recognition.infer import Predictor
from ssl_recognition.landmarks import N_HAND, HolisticExtractor, _to_array

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
CERTS = ROOT / "certs"

app = FastAPI(title="Saudi Sign Language")
STATE: dict = {"run": None, "names": None, "window": 40, "stride": 1}


def load_names(path: Path | None, labels: list[str]) -> list[dict]:
    table = {}
    if path and Path(path).exists():
        table = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for lab in labels:
        sid = lab.split()[0]
        row = table.get(sid, {})
        out.append({"id": sid, "ar": row.get("ar", ""), "en": row.get("en", sid)})
    return out


def predictor() -> Predictor:
    if "predictor" not in STATE:
        STATE["predictor"] = Predictor(Path(STATE["run"]), use_onnx=True)
        STATE["labels"] = load_names(STATE["names"], STATE["predictor"].labels)
    return STATE["predictor"]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/labels")
def labels():
    predictor()
    return JSONResponse(STATE["labels"])


def _reference_file(sid: str, suffix: str, folder: str) -> Path | None:
    """First reference clip for a sign: prefer signer 01 test clips, then anything."""
    d = Path(folder) / sid
    if not d.is_dir():
        return None
    files = sorted(d.glob(f"s01_test_*{suffix}")) or sorted(d.glob(f"*{suffix}"))
    return files[0] if files else None


@app.get("/api/ref/{sid}")
def reference(sid: str):
    """Landmark sequence of one reference clip, for the skeleton animation in the UI."""
    f = _reference_file(sid, ".npz", "data/landmarks")
    if f is None:
        return JSONResponse({"error": "no reference clip"}, status_code=404)
    with np.load(f) as z:
        x = z["x"]
    frames = []
    for row in x:
        pose = row[:27].reshape(9, 3)[:, :2]
        left = row[27:90].reshape(21, 3)[:, :2]
        right = row[90:153].reshape(21, 3)[:, :2]
        frames.append(
            {
                "pose": np.round(pose, 3).tolist(),
                "left": np.round(left, 3).tolist() if row[153] > 0 else [],
                "right": np.round(right, 3).tolist() if row[154] > 0 else [],
            }
        )
    return JSONResponse({"id": sid, "fps": 30, "frames": frames})


SPRITE_FRAMES, SPRITE_H = 24, 180


@app.get("/api/sprite/{sid}")
def sprite(sid: str):
    """Reference clip rendered once into a horizontal strip of JPEG frames (codec-free playback)."""
    dest = Path("data/sprites") / f"{sid}.jpg"
    if not dest.exists():
        f = _reference_file(sid, ".mp4", "data/videos")
        if f is None:
            return JSONResponse({"error": "no video"}, status_code=404)
        cap = cv2.VideoCapture(str(f))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return JSONResponse({"error": "unreadable video"}, status_code=500)
        idx = np.linspace(0, len(frames) - 1, SPRITE_FRAMES).astype(int)
        fh, fw = frames[0].shape[:2]
        side = min(fh, fw)  # centre square crop: the signer stands in the middle of the frame
        x0, y0 = (fw - side) // 2, (fh - side) // 2
        strip = np.concatenate(
            [
                cv2.resize(
                    frames[i][y0 : y0 + side, x0 : x0 + side],
                    (SPRITE_H, SPRITE_H),
                    interpolation=cv2.INTER_AREA,
                )
                for i in idx
            ],
            axis=1,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), strip, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return FileResponse(dest, media_type="image/jpeg", headers={"X-Frames": str(SPRITE_FRAMES)})


@app.get("/api/clip/{sid}")
def clip(sid: str):
    """Reference video (H.264 mp4) of one sign, when the KArSL videos are present locally."""
    f = _reference_file(sid, ".mp4", "data/videos")
    if f is None:
        return JSONResponse({"error": "no video"}, status_code=404)
    return FileResponse(f, media_type="video/mp4")


def landmarks_payload(result) -> dict:
    def pts(lms, n):
        arr = _to_array(lms, n)
        return [] if arr is None else np.round(arr[:, :2], 4).tolist()

    return {
        "pose": pts(result.pose_landmarks, 33) if result else [],
        "left": pts(result.left_hand_landmarks, N_HAND) if result else [],
        "right": pts(result.right_hand_landmarks, N_HAND) if result else [],
    }


@app.get("/models/holistic_landmarker.task")
def model_file():
    """Serve the MediaPipe model so the browser can run landmarking itself (fast path)."""
    from ssl_recognition.landmarks import DEFAULT_MODEL

    return FileResponse(DEFAULT_MODEL, media_type="application/octet-stream")


class Segmenter:
    """Cuts the stream into single signs. A sign starts when hands appear and ends when no hand
    has been seen for ``gap_ms`` (time-based, so it behaves the same at 10 or 30 fps), or when the
    user presses Done. Short drop-outs inside a sign (e.g. between the two movements of "skull")
    are kept inside the sign."""

    def __init__(self, gap_ms: float = 700, min_len: int = 8, max_len: int = 180):
        self.gap_ms, self.min_len, self.max_len = gap_ms, min_len, max_len
        self.seg: list[np.ndarray] = []
        self.last_hand_ms: float | None = None

    def push(self, feat: np.ndarray, t_ms: float) -> np.ndarray | None:
        if feat[-2] + feat[-1] > 0:
            self.seg.append(feat)
            self.last_hand_ms = t_ms
            if len(self.seg) >= self.max_len:
                return self.flush()
        elif self.seg:
            self.seg.append(feat)  # keep the gap frames so timing inside the sign is preserved
            if self.last_hand_ms is not None and t_ms - self.last_hand_ms >= self.gap_ms:
                return self.flush()
        return None

    def flush(self) -> np.ndarray | None:
        seg, self.seg, self.last_hand_ms = self.seg, [], None
        # trim trailing no-hand frames
        while seg and seg[-1][-2] + seg[-1][-1] <= 0:
            seg.pop()
        return np.stack(seg) if len(seg) >= self.min_len else None


def top3(pred, names, x) -> list[dict]:
    p = pred.predict(x)
    return [{**names[i], "p": round(float(p[i]), 3)} for i in np.argsort(-p)[:3]]


# --- phone-as-camera, laptop-as-display relay -------------------------------------------------
DISPLAYS: set[WebSocket] = set()
RELAY: dict = {"end": False}


async def broadcast_text(msg: dict) -> None:
    dead = []
    data = json.dumps(msg)
    for d in DISPLAYS:
        try:
            await d.send_text(data)
        except Exception:  # noqa: BLE001 - a closed display must not break the camera
            dead.append(d)
    for d in dead:
        DISPLAYS.discard(d)


async def broadcast_bytes(data: bytes) -> None:
    dead = []
    for d in DISPLAYS:
        try:
            await d.send_bytes(data)
        except Exception:  # noqa: BLE001
            dead.append(d)
    for d in dead:
        DISPLAYS.discard(d)


class Session:
    """Per-connection state: rolling window for live guesses + segmenter for final answers."""

    def __init__(self, gap_ms: float = 700):
        self.pred = predictor()
        self.names = STATE["labels"]
        self.buf: collections.deque = collections.deque(maxlen=STATE["window"])
        self.seg = Segmenter(gap_ms=gap_ms)
        self.frame_i = 0
        self.top: list[dict] = []

    def step(self, feat: np.ndarray, t_ms: float, end: bool = False) -> dict:
        self.buf.append(feat)
        self.frame_i += 1
        if self.frame_i % STATE["stride"] == 0 and len(self.buf) >= STATE["window"] // 2:
            x = np.stack(self.buf)
            ok = (x[:, -2] + x[:, -1] > 0).mean() > 0.5
            self.top = top3(self.pred, self.names, x) if ok else []
        if RELAY["end"]:
            RELAY["end"], end = False, True
        done = self.seg.flush() if end else self.seg.push(feat, t_ms)
        out = {"top": self.top, "hands": int(feat[-2] + feat[-1]), "recording": len(self.seg.seg)}
        if done is not None:
            out["final"] = top3(self.pred, self.names, done)
            out["final_frames"] = int(len(done))
        return out


@app.websocket("/ws_feat")
async def ws_feat(websocket: WebSocket):
    """Fast path: the browser runs MediaPipe and sends feature rows (text). It may also send small
    JPEG preview frames (binary), which are relayed to laptop displays."""
    await websocket.accept()
    sess = Session()
    try:
        while True:
            m = await websocket.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                await broadcast_bytes(m["bytes"])
                continue
            msg = json.loads(m.get("text") or "{}")
            if "gap_ms" in msg:
                sess.seg.gap_ms = float(msg["gap_ms"])
            t = time.perf_counter()
            feat = np.asarray(msg.get("f", []), dtype=np.float32)
            if feat.shape != (155,):
                feat = np.zeros(155, np.float32)
            t_ms = float(msg.get("t", time.time() * 1000))
            out = sess.step(feat, t_ms, end=bool(msg.get("end")))
            out["ms"] = round((time.perf_counter() - t) * 1000, 1)
            await websocket.send_text(json.dumps(out))
            if DISPLAYS:
                await broadcast_text({**out, "lm": msg.get("lm"), "src": "phone"})
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Server path: the browser sends JPEG frames; MediaPipe runs here."""
    await websocket.accept()
    sess = Session()
    t0 = time.time()
    with HolisticExtractor() as ex:
        try:
            while True:
                m = await websocket.receive()
                if m.get("type") == "websocket.disconnect":
                    break
                if m.get("text") is not None:
                    msg = json.loads(m["text"])
                    if "gap_ms" in msg:
                        sess.seg.gap_ms = float(msg["gap_ms"])
                    if not msg.get("end"):
                        continue
                    out = sess.step(np.zeros(155, np.float32), time.time() * 1000, end=True)
                    out["lm"] = {"pose": [], "left": [], "right": []}
                else:
                    data = m["bytes"]
                    t = time.perf_counter()
                    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    feat = ex.process(frame, int((time.time() - t0) * 1000))
                    out = sess.step(feat, time.time() * 1000)
                    out["lm"] = landmarks_payload(ex.last_result)
                    out["ms"] = round((time.perf_counter() - t) * 1000, 1)
                    if DISPLAYS:
                        await broadcast_bytes(data)
                await websocket.send_text(json.dumps(out))
                if DISPLAYS:
                    await broadcast_text({**out, "src": "phone"})
        except WebSocketDisconnect:
            pass


@app.websocket("/ws_display")
async def ws_display(websocket: WebSocket):
    """Laptop display: receives results and preview frames from whichever device is the camera."""
    await websocket.accept()
    DISPLAYS.add(websocket)
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            if msg.get("end"):
                RELAY["end"] = True  # Done pressed on the laptop ends the phone's current sign
    except WebSocketDisconnect:
        pass
    finally:
        DISPLAYS.discard(websocket)


@app.get("/camera")
def camera_page():
    return FileResponse(STATIC / "camera.html")


@app.get("/api/info")
def info():
    port = STATE.get("https_port", 8443)
    return {"camera_url": f"https://{lan_ip()}:{port}/camera", "displays": len(DISPLAYS)}


def ensure_cert() -> tuple[Path, Path]:
    """Create a self-signed certificate once so phones can open the camera over HTTPS."""
    CERTS.mkdir(exist_ok=True)
    crt, key = CERTS / "cert.pem", CERTS / "key.pem"
    if crt.exists() and key.exists():
        return crt, key
    import datetime as dt
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "saudi-sign-language.local")])
    sans = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        sans.append(x509.IPAddress(ipaddress.ip_address(lan_ip())))
    except ValueError:
        pass
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(k.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .sign(k, hashes.SHA256())
    )
    key.write_bytes(
        k.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return crt, key


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    import asyncio

    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/final_tf_all_official")
    ap.add_argument("--names", default="data/karsl_labels.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--https-port", type=int, default=8443)
    ap.add_argument("--no-https", action="store_true", help="laptop only, no phone")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1, help="predict every N frames (1 = every)")
    args = ap.parse_args()
    STATE.update(
        run=args.run,
        names=args.names,
        window=args.window,
        stride=args.stride,
        https_port=args.https_port,
    )
    predictor()  # load once, fail early

    # One process serves both ports so the phone (HTTPS) and the laptop display (HTTP) share
    # the same relay state.
    servers = [
        uvicorn.Server(
            uvicorn.Config(app, host=args.host, port=args.http_port, log_level="warning")
        )
    ]
    if not args.no_https:
        crt, key = ensure_cert()
        servers.append(
            uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=args.host,
                    port=args.https_port,
                    log_level="warning",
                    ssl_certfile=str(crt),
                    ssl_keyfile=str(key),
                )
            )
        )
    ip = lan_ip()
    print()
    print(f"  Laptop:          http://localhost:{args.http_port}")
    if not args.no_https:
        print(f"  Phone camera:    https://{ip}:{args.https_port}/camera")
        print(f"  Phone full app:  https://{ip}:{args.https_port}")
    print()

    async def serve_all():
        await asyncio.gather(*(srv.serve() for srv in servers))

    asyncio.run(serve_all())


if __name__ == "__main__":
    main()
