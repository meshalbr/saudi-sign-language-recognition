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
STATE: dict = {"run": None, "names": None, "window": 40, "stride": 4}


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


def landmarks_payload(result) -> dict:
    def pts(lms, n):
        arr = _to_array(lms, n)
        return [] if arr is None else np.round(arr[:, :2], 4).tolist()

    return {
        "pose": pts(result.pose_landmarks, 33) if result else [],
        "left": pts(result.left_hand_landmarks, N_HAND) if result else [],
        "right": pts(result.right_hand_landmarks, N_HAND) if result else [],
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    pred = predictor()
    names = STATE["labels"]
    buf: collections.deque = collections.deque(maxlen=STATE["window"])
    frame_i = 0
    top: list[dict] = []
    t0 = time.time()
    with HolisticExtractor() as ex:
        try:
            while True:
                data = await websocket.receive_bytes()
                t = time.perf_counter()
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                feat = ex.process(frame, int((time.time() - t0) * 1000))
                buf.append(feat)
                frame_i += 1
                hands = int(feat[-2] + feat[-1])
                if frame_i % STATE["stride"] == 0 and len(buf) >= STATE["window"] // 2:
                    x = np.stack(buf)
                    if (x[:, -2] + x[:, -1] > 0).mean() > 0.5:
                        p = pred.predict(x)
                        order = np.argsort(-p)[:3]
                        top = [{**names[i], "p": round(float(p[i]), 3)} for i in order]
                    else:
                        top = []
                await websocket.send_text(
                    json.dumps(
                        {
                            "top": top,
                            "hands": hands,
                            "lm": landmarks_payload(ex.last_result),
                            "ms": round((time.perf_counter() - t) * 1000, 1),
                        }
                    )
                )
        except WebSocketDisconnect:
            pass


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
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/final_tf_all_official")
    ap.add_argument("--names", default="data/karsl_labels.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--https", action="store_true", help="self-signed TLS so phones allow the camera"
    )
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    STATE.update(run=args.run, names=args.names, window=args.window, stride=args.stride)
    predictor()  # load once, fail early
    port = args.port or (8443 if args.https else 8000)
    kw = {}
    if args.https:
        crt, key = ensure_cert()
        kw = {"ssl_certfile": str(crt), "ssl_keyfile": str(key)}
    scheme = "https" if args.https else "http"
    print(f"\n  Laptop:  {scheme}://localhost:{port}\n  Phone:   {scheme}://{lan_ip()}:{port}\n")
    uvicorn.run(app, host=args.host, port=port, log_level="warning", **kw)


if __name__ == "__main__":
    main()
