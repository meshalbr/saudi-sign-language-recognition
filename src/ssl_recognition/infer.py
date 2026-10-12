"""Live webcam recognition on CPU.

    python -m ssl_recognition.infer --run runs/baseline            # PyTorch checkpoint
    python -m ssl_recognition.infer --run runs/baseline --onnx      # ONNX Runtime

Keeps a sliding window of the last ``window`` frames, resamples it to the model's sequence
length and predicts every ``stride`` frames while at least one hand is visible.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np

from .data import load_labels, resample
from .landmarks import HolisticExtractor, draw_result


class Predictor:
    def __init__(self, run_dir: Path, use_onnx: bool = False):
        self.labels = load_labels(run_dir / "labels.json")
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
        x = resample(x, self.seq_len)[None]
        if self.use_onnx:
            logits = self.sess.run(None, {self.input_name: x})[0][0]
        else:
            with self._torch.no_grad():
                logits = self.model(self._torch.from_numpy(x))[0].numpy()
        z = logits - logits.max()
        p = np.exp(z)
        return p / p.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    pred = Predictor(Path(args.run), use_onnx=args.onnx)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")

    buf: collections.deque = collections.deque(maxlen=args.window)
    top = []
    frame_i = 0
    fps_t, fps_n, fps = time.time(), 0, 0.0
    t_start = time.time()
    with HolisticExtractor() as ex:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            feat = ex.process(frame, int((time.time() - t_start) * 1000))
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
            for k, (lab, prob) in enumerate(top):
                colour = (0, 255, 0) if (k == 0 and prob >= args.threshold) else (200, 200, 200)
                cv2.putText(
                    frame,
                    f"{lab}: {prob:.2f}",
                    (10, 40 + 30 * k),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    colour,
                    2,
                )
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
            cv2.putText(
                frame,
                f"{fps:.1f} FPS  {'onnx' if args.onnx else 'torch'}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Saudi Sign Language", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
