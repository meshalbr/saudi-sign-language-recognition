"""Live webcam recognition on CPU.

    python -m ssl_recognition.infer --run runs/baseline            # PyTorch checkpoint
    python -m ssl_recognition.infer --run runs/baseline --onnx      # ONNX Runtime

Keeps a sliding window of the last ``window`` frames, resamples it to the model's sequence
length and predicts every ``stride`` frames while at least one hand is visible.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .data import load_labels, resample
from .features import fill_gaps, transform
from .landmarks import HolisticExtractor, draw_result


class Predictor:
    def __init__(self, run_dir: Path, use_onnx: bool = False, names: Path | None = None):
        self.labels = load_labels(run_dir / "labels.json")
        cfg_path = run_dir / "config.json"
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        if names is not None and Path(names).exists():
            # Optional display names, e.g. data/karsl_labels.json: {"0071": {"ar":..., "en":...}}
            table = json.loads(Path(names).read_text(encoding="utf-8"))
            self.labels = [
                f"{lab} {table[lab]['en']}" if lab in table and "en" in table[lab] else lab
                for lab in self.labels
            ]
        self.use_onnx = use_onnx
        if use_onnx:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(
                str(run_dir / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            self.input_name = self.sess.get_inputs()[0].name
            self.seq_len = int(self.sess.get_inputs()[0].shape[1])
        else:
            import torch

            from .model import build_model

            ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
            self.model = build_model(ckpt["cfg"], ckpt["num_classes"])
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.seq_len = int(ckpt["cfg"].get("seq_len", 32))
            self._torch = torch

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (T, D) -> softmax probabilities over labels."""
        if self.cfg.get("fill_gaps"):
            x = fill_gaps(x, int(self.cfg["fill_gaps"]))
        x = transform(resample(x, self.seq_len), self.cfg)[None]
        if self.use_onnx:
            logits = self.sess.run(None, {self.input_name: x})[0][0]
        else:
            with self._torch.no_grad():
                logits = self.model(self._torch.from_numpy(x))[0].numpy()
        z = logits - logits.max()
        p = np.exp(z)
        return p / p.sum()


WIN = "Saudi Sign Language"
PANEL_W = 460
INK, PANEL_BG, MUTED, GREEN, AMBER, WHITE = (
    (30, 24, 20),
    (36, 30, 26),
    (170, 170, 170),
    (90, 200, 110),
    (60, 160, 230),
    (240, 240, 240),
)


def open_camera(source: str) -> cv2.VideoCapture:
    """Open a local camera index or a network stream (phone camera apps give an http/rtsp URL)."""
    if source.isdigit():
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def _text(img, s, xy, scale, colour, thick=2):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def compose(frame, top, threshold, hands_now, fps, use_onnx) -> np.ndarray:
    """Video on the left at 720p height, dark results panel on the right; scales with the window."""
    h = 720
    w = int(frame.shape[1] * h / frame.shape[0])
    video = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    panel = np.full((h, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    _text(panel, "SAUDI SIGN LANGUAGE", (24, 48), 0.8, MUTED, 2)
    _text(panel, "Top prediction", (24, 110), 0.6, MUTED, 1)
    if top:
        lab, prob = top[0]
        sid, _, name = lab.partition(" ")
        colour = GREEN if prob >= threshold else AMBER
        _text(panel, name[:22] or sid, (24, 165), 1.3, WHITE, 3)
        _text(panel, f"sign {sid}", (24, 200), 0.7, MUTED, 1)
        cv2.rectangle(panel, (24, 220), (PANEL_W - 24, 244), (60, 54, 50), -1)
        cv2.rectangle(panel, (24, 220), (24 + int((PANEL_W - 48) * prob), 244), colour, -1)
        _text(panel, f"{prob * 100:.0f}% confidence", (24, 275), 0.7, colour, 2)
        _text(panel, "Also likely", (24, 340), 0.6, MUTED, 1)
        for k, (lab2, p2) in enumerate(top[1:3]):
            sid2, _, name2 = lab2.partition(" ")
            _text(panel, f"{name2[:20] or sid2}", (24, 380 + 44 * k), 0.8, WHITE, 2)
            _text(panel, f"{p2 * 100:.0f}%", (PANEL_W - 90, 380 + 44 * k), 0.8, MUTED, 2)
    else:
        _text(panel, "Show your hands", (24, 165), 1.1, AMBER, 2)
        _text(panel, "to the camera", (24, 210), 1.1, AMBER, 2)
    _text(panel, f"hands detected: {hands_now}", (24, h - 110), 0.6, MUTED, 1)
    _text(
        panel,
        f"{fps:.1f} FPS  {'ONNX Runtime' if use_onnx else 'PyTorch'}",
        (24, h - 80),
        0.6,
        MUTED,
        1,
    )
    _text(panel, "Q / Esc / close window to quit", (24, h - 40), 0.55, MUTED, 1)
    return np.concatenate([video, panel], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument(
        "--camera",
        default="0",
        help="camera index (0, 1, ...) or a stream URL from a phone camera app",
    )
    ap.add_argument("--names", default="data/karsl_labels.json", help="label display names")
    args = ap.parse_args()

    pred = Predictor(Path(args.run), use_onnx=args.onnx, names=Path(args.names))
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera!r}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)  # resizable; drag the corner or maximise
    cv2.resizeWindow(WIN, 1280, 720)

    buf: collections.deque = collections.deque(maxlen=args.window)
    top: list[tuple[str, float]] = []
    frame_i = 0
    fps_t, fps_n, fps = time.time(), 0, 0.0
    t_start = time.time()
    hands_now = 0
    with HolisticExtractor() as ex:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            feat = ex.process(frame, int((time.time() - t_start) * 1000))
            hands_now = int(feat[-2] + feat[-1])
            buf.append(feat)
            frame_i += 1
            if frame_i % args.stride == 0 and len(buf) >= args.window // 2:
                x = np.stack(buf)
                if (x[:, -2] + x[:, -1] > 0).mean() > 0.5:
                    p = pred.predict(x)
                    order = np.argsort(-p)[:3]
                    top = [(pred.labels[i], float(p[i])) for i in order]
                else:
                    top = []
            draw_result(frame, ex.last_result)
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
            canvas = compose(frame, top, args.threshold, hands_now, fps, args.onnx)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:  # window closed with X
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
