"""Export a trained checkpoint to ONNX and check it against PyTorch.

python scripts/export_onnx.py --run runs/baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ssl_recognition.landmarks import FEATURE_DIM
from ssl_recognition.model import build_model


def export(run_dir: Path) -> Path:
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    model = build_model(ckpt["cfg"], ckpt["num_classes"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    seq_len = int(ckpt["cfg"].get("seq_len", 32))
    dummy = torch.zeros(1, seq_len, FEATURE_DIM)
    dest = run_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(dest),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randn(2, seq_len, FEATURE_DIM).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = sess.run(None, {"features": x})[0]
    err = float(np.abs(ref - out).max())
    print(f"exported {dest} ({dest.stat().st_size / 1e6:.2f} MB), max |torch - onnx| = {err:.2e}")
    assert err < 1e-3, "ONNX output does not match PyTorch"
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/baseline")
    export(Path(ap.parse_args().run))
