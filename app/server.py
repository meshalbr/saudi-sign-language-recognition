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
import asyncio
import collections
import json
import socket
import threading
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


_PRED_LOCK = threading.Lock()


def predictor() -> Predictor:
    with _PRED_LOCK:
        return _predictor_locked()


def _predictor_locked() -> Predictor:
    if "predictor" not in STATE:
        STATE["predictor"] = Predictor(Path(STATE["run"]), use_onnx=True)
        STATE["labels"] = load_names(STATE["names"], STATE["predictor"].labels)
    return STATE["predictor"]


NO_CACHE = {"Cache-Control": "no-store"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.get("/api/labels")
def labels():
    predictor()
    return JSONResponse(STATE["labels"])


def _valid_sid(sid: str) -> bool:
    predictor()
    return sid in {lab["id"] for lab in STATE["labels"]}


def _coverage(npz: Path) -> float:
    with np.load(npz) as z:
        x = z["x"]
    present = (x[:, -2] + x[:, -1]) > 0
    return float(present.mean())


_BEST_REF: dict[str, str] = {}


def _best_reference_stem(sid: str) -> str | None:
    """The reference recording with the best hand tracking among signer 01 test clips."""
    if sid not in _BEST_REF:
        cands = sorted(Path("data/landmarks", sid).glob("s01_test_*.npz")) or sorted(
            Path("data/landmarks", sid).glob("*.npz")
        )
        if not cands:
            return None
        _BEST_REF[sid] = max(cands[:20], key=_coverage).stem
    return _BEST_REF[sid]


def _reference_file(sid: str, suffix: str, folder: str) -> Path | None:
    stem = _best_reference_stem(sid)
    if stem is not None:
        f = Path(folder) / sid / f"{stem}{suffix}"
        if f.exists():
            return f
    d = Path(folder) / sid
    files = sorted(d.glob(f"*{suffix}")) if d.is_dir() else []
    return files[0] if files else None


@app.get("/api/ref/{sid}")
def reference(sid: str):
    """Landmark sequence of one reference clip, for the skeleton animation in the UI."""
    if not _valid_sid(sid):
        return JSONResponse({"error": "unknown sign"}, status_code=404)
    f = _reference_file(sid, ".npz", "data/landmarks")
    if f is None:
        return JSONResponse({"error": "no reference clip"}, status_code=404)
    from ssl_recognition.features import fill_gaps

    with np.load(f) as z:
        x = fill_gaps(z["x"], 8)
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
    if not _valid_sid(sid):
        return JSONResponse({"error": "unknown sign"}, status_code=404)
    dest = Path("data/sprites") / f"{sid}_{_best_reference_stem(sid)}.jpg"
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
    if not _valid_sid(sid):
        return JSONResponse({"error": "unknown sign"}, status_code=404)
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


LAST_SIGN: dict = {"x": None}
REST_Y = 1.6  # shoulder widths below the shoulder line; KArSL hands start above ~1.3, hips ~2.5
WRIST_Y = (16, 19)  # feature indices of the two wrists' y (pose block positions 5 and 6)


def is_active(feat: np.ndarray, rest_y: float = REST_Y) -> bool:
    """Signing = at least one hand detected and a wrist raised above the rest line."""
    if feat[-2] + feat[-1] <= 0 or not feat[:27].any():
        return False
    return min(feat[WRIST_Y[0]], feat[WRIST_Y[1]]) < rest_y


STILL_MS = 1000  # hands held still this long finish the sign (KArSL: ~2% of signs cut early)
STILL_THR = 0.035  # max hand-centre movement during the hold, in shoulder widths
L_HAND, R_HAND = slice(27, 90), slice(90, 153)


def hand_centres(feat: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    left = feat[L_HAND].reshape(21, 3)[:, :2].mean(0) if feat[153] > 0 else None
    right = feat[R_HAND].reshape(21, 3)[:, :2].mean(0) if feat[154] > 0 else None
    return left, right


class Segmenter:
    """Cuts the stream into single signs.

    A sign starts when a wrist rises above the rest line (so raising your hand from your lap is
    not part of the sign, matching how KArSL clips begin). It ends when any of these happens:
    the hands are held still for ``still_ms``; both hands have been below the line or out of
    view for ``gap_ms``; the sign reaches ``max_ms``; or the user presses Done. Short drop-outs
    inside a sign, such as the pause between the two movements of "skull", stay inside it."""

    def __init__(
        self,
        gap_ms: float = 700,
        min_len: int = 8,
        max_ms: float = 6000,
        rest_y=REST_Y,
        still_ms: float | None = STILL_MS,
        still_thr: float = STILL_THR,
        min_sign_ms: float = 800,
    ):
        self.gap_ms, self.min_len, self.max_ms, self.rest_y = gap_ms, min_len, max_ms, rest_y
        self.still_ms, self.still_thr, self.min_sign_ms = still_ms, still_thr, min_sign_ms
        self.seg: list[np.ndarray] = []
        self.times: list[float] = []
        self.start_ms: float | None = None
        self.last_active_ms: float | None = None
        self.last: np.ndarray | None = None  # last completed sign
        self.still_progress = 0.0  # 0..1, shown in the UI while holding still
        self.end_reason = ""

    def _still_run(self) -> tuple[float, int]:
        """Length in ms of the trailing run in which every visible hand stayed within
        ``still_thr``, and the index where that run starts."""
        lo = {0: None, 1: None}
        hi = {0: None, 1: None}
        start = len(self.seg)
        for i in range(len(self.seg) - 1, -1, -1):
            if not is_active(self.seg[i], self.rest_y):
                break
            cs = hand_centres(self.seg[i])
            moved = False
            for h, c in enumerate(cs):
                if c is None:
                    continue
                lo[h] = c if lo[h] is None else np.minimum(lo[h], c)
                hi[h] = c if hi[h] is None else np.maximum(hi[h], c)
                if np.linalg.norm(hi[h] - lo[h]) > self.still_thr:
                    moved = True
            if moved:
                break
            start = i
        if start >= len(self.seg):
            return 0.0, start
        return self.times[-1] - self.times[start], start

    def push(self, feat: np.ndarray, t_ms: float) -> np.ndarray | None:
        self.still_progress = 0.0
        if is_active(feat, self.rest_y):
            if not self.seg:
                self.start_ms = t_ms
            self.seg.append(feat)
            self.times.append(t_ms)
            self.last_active_ms = t_ms
            if self.start_ms is not None and t_ms - self.start_ms >= self.max_ms:
                return self.flush("time limit")
            if self.still_ms and t_ms - self.start_ms >= self.min_sign_ms:
                run_ms, run_start = self._still_run()
                self.still_progress = min(1.0, run_ms / self.still_ms)
                if run_ms >= self.still_ms:
                    # keep a short part of the final hold, drop the rest of the waiting time
                    keep_until = self.times[run_start] + 300
                    cut = next(
                        (i for i, t in enumerate(self.times) if t > keep_until), len(self.seg)
                    )
                    self.seg, self.times = (
                        self.seg[: max(cut, run_start + 1)],
                        self.times[: max(cut, run_start + 1)],
                    )
                    return self.flush("hands still")
        elif self.seg:
            self.seg.append(feat)
            self.times.append(t_ms)
            if self.last_active_ms is not None and t_ms - self.last_active_ms >= self.gap_ms:
                return self.flush("hands lowered")
        return None

    def flush(self, reason: str = "done pressed") -> np.ndarray | None:
        seg, self.seg, self.times = self.seg, [], []
        self.last_active_ms, self.start_ms, self.still_progress = None, None, 0.0
        while seg and not is_active(seg[-1], self.rest_y):  # trim the lowering at the end
            seg.pop()
        if len(seg) < self.min_len:
            return None
        self.end_reason = reason
        self.last = np.stack(seg)
        return self.last


def top3(pred, names, x) -> list[dict]:
    p = pred.predict(x)
    return [{**names[i], "p": round(float(p[i]), 3)} for i in np.argsort(-p)[:3]]


WINDOW_MARGIN = 0.15  # a shorter window must beat the full sign by this much confidence


def best_window(
    pred, names, x: np.ndarray, margin: float | None = None
) -> tuple[list[dict], tuple[int, int]]:
    """Classify the sub-window of a completed sign that the model is most sure about.

    People raise their hand to the start position before a sign and lower it afterwards; KArSL
    clips contain neither. Trying start points in the first part of the segment and end points
    in the last part, and keeping the most confident window, drops that travel automatically.
    """
    n = len(x)
    if n < 16:
        return top3(pred, names, x), (0, n)
    min_len = max(12, int(n * 0.4))
    starts = sorted({int(v) for v in np.linspace(0, int(n * 0.45), 10)})
    ends = sorted({int(v) for v in np.linspace(int(n * 0.65), n, 7)})
    margin = WINDOW_MARGIN if margin is None else margin
    full = pred.predict(x)
    best, best_conf, span = full, float(full.max()) + margin, (0, n)
    for s0 in starts:
        for e0 in ends:
            if e0 - s0 < min_len:
                continue
            p = pred.predict(x[s0:e0])
            if p.max() > best_conf:
                best, best_conf, span = p, float(p.max()), (s0, e0)
    order = np.argsort(-best)[:3]
    return [{**names[i], "p": round(float(best[i]), 3)} for i in order], span


# --- phone-as-camera, laptop-as-display relay -------------------------------------------------
DISPLAYS: set[WebSocket] = set()
RELAY: dict = {"end": False}


async def broadcast_text(msg: dict) -> None:
    dead = []
    data = json.dumps(msg)
    for d in list(DISPLAYS):
        try:
            await d.send_text(data)
        except Exception:  # noqa: BLE001 - a closed display must not break the camera
            dead.append(d)
    for d in dead:
        DISPLAYS.discard(d)


async def broadcast_bytes(data: bytes) -> None:
    dead = []
    for d in list(DISPLAYS):
        try:
            await d.send_bytes(data)
        except Exception:  # noqa: BLE001
            dead.append(d)
    for d in dead:
        DISPLAYS.discard(d)


class Session:
    """Per-connection state: rolling window for live guesses + segmenter for final answers."""

    def __init__(self, gap_ms: float = 700, relay: bool = False):
        predictor()
        self.relay = relay
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
            self.top = top3(predictor(), STATE["labels"], x) if ok else []
        if self.relay and RELAY["end"]:  # Done pressed on the laptop display
            RELAY["end"], end = False, True
        done = self.seg.flush() if end else self.seg.push(feat, t_ms)
        out = {
            "top": self.top,
            "hands": int(feat[-2] + feat[-1]),
            "recording": len(self.seg.seg),
            "still": round(self.seg.still_progress, 2),
        }
        if done is not None:
            final, span = best_window(predictor(), STATE["labels"], done)
            out["final"] = final
            out["final_frames"] = int(span[1] - span[0])
            out["final_span"] = [int(span[0]), int(span[1]), int(len(done))]
            out["end_reason"] = self.seg.end_reason
            LAST_SIGN["x"] = done[span[0] : span[1]]  # what teach mode saves
        return out


@app.websocket("/ws_feat")
async def ws_feat(websocket: WebSocket):
    """Fast path: the browser runs MediaPipe and sends feature rows (text). It may also send small
    JPEG preview frames (binary), which are relayed to laptop displays."""
    await websocket.accept()
    relay = websocket.query_params.get("role") == "camera"
    sess = Session(relay=relay)
    try:
        while True:
            m = await websocket.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                if relay:
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
            out = await asyncio.to_thread(sess.step, feat, t_ms, bool(msg.get("end")))
            out["ms"] = round((time.perf_counter() - t) * 1000, 1)
            await websocket.send_text(json.dumps(out))
            if relay and DISPLAYS:
                await broadcast_text({**out, "lm": msg.get("lm"), "src": "phone"})
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Server path: the browser sends JPEG frames; MediaPipe runs here."""
    await websocket.accept()
    relay = websocket.query_params.get("role") == "camera"
    sess = Session(relay=relay)
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
                    out = await asyncio.to_thread(
                        sess.step, np.zeros(155, np.float32), time.time() * 1000, True
                    )
                    out["lm"] = {"pose": [], "left": [], "right": []}
                else:
                    data = m["bytes"]
                    t = time.perf_counter()

                    def work(data=data):
                        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            return None
                        feat = ex.process(frame, int((time.time() - t0) * 1000))
                        res = sess.step(feat, time.time() * 1000)
                        res["lm"] = landmarks_payload(ex.last_result)
                        return res

                    out = await asyncio.to_thread(work)
                    if out is None:
                        continue
                    out["ms"] = round((time.perf_counter() - t) * 1000, 1)
                    if relay and DISPLAYS:
                        await broadcast_bytes(data)
                await websocket.send_text(json.dumps(out))
                if relay and DISPLAYS:
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


@app.post("/api/save/{sid}")
def save_last(sid: str):
    """Store the most recent completed sign as the user's example of word ``sid``."""
    from app import personalise

    if LAST_SIGN["x"] is None:
        return JSONResponse({"error": "no sign recorded yet"}, status_code=400)
    if sid not in {lab["id"] for lab in STATE["labels"]}:
        return JSONResponse({"error": "unknown sign"}, status_code=404)
    n = personalise.save_example(sid, LAST_SIGN["x"])
    LAST_SIGN["x"] = None  # one sign is saved once
    return {"id": sid, "count": n}


@app.get("/api/user")
def user_info():
    from app import personalise

    return {
        "counts": personalise.counts(),
        "status": personalise.STATUS,
        "model": "personal" if "personal" in str(STATE["run"]) else "base",
    }


@app.post("/api/personalise")
def personalise_start():
    from app import personalise

    base = Path(STATE.get("base_run") or STATE["run"])
    if "personal" in str(base):
        base = Path("runs/final_v2")
    STATE["base_run"] = str(base)

    def swap(run: Path):
        STATE["run"] = str(run)
        STATE.pop("predictor", None)
        predictor()

    ok = personalise.start(base, Path("runs/personal"), swap)
    return {"started": ok, "status": personalise.STATUS}


@app.post("/api/model/{which}")
def choose_model(which: str):
    """Switch between the base model and the personalised one."""
    base = STATE.get("base_run") or STATE["run"]
    STATE["base_run"] = base
    target = "runs/personal" if which == "personal" else base
    if which == "personal" and not (Path(target) / "model.onnx").exists():
        return JSONResponse({"error": "no personal model yet"}, status_code=404)
    STATE["run"] = target
    STATE.pop("predictor", None)
    predictor()
    return {"model": which}


@app.get("/camera")
def camera_page():
    return FileResponse(STATIC / "camera.html", headers=NO_CACHE)


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
    ap.add_argument("--run", default="runs/final_v2")
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
