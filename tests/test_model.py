import numpy as np
import torch

from ssl_recognition.data import save_clip
from ssl_recognition.landmarks import FEATURE_DIM
from ssl_recognition.model import GRUClassifier, count_params
from ssl_recognition.train import train


def test_forward_shape():
    m = GRUClassifier(num_classes=7, hidden=16, layers=1)
    out = m(torch.zeros(3, 32, FEATURE_DIM))
    assert out.shape == (3, 7)
    assert count_params(m) > 0


def test_train_smoke_on_tiny_synthetic_dataset(tmp_path):
    rng = np.random.RandomState(0)
    for lab, offset in (("a", 0.0), ("b", 1.0)):
        for s in ("s1", "s2"):
            for i in range(4):
                x = (rng.rand(24, FEATURE_DIM) + offset).astype(np.float32)
                save_clip(tmp_path / "landmarks" / lab / f"{s}_{i:03d}.npz", x, lab, s, 30.0)
    cfg = {
        "data_dir": str(tmp_path / "landmarks"),
        "seq_len": 16,
        "batch_size": 4,
        "epochs": 3,
        "hidden": 16,
        "layers": 1,
        "out_dir": str(tmp_path / "run"),
        "augment": True,
    }
    result = train(cfg, test_signer="s2")
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "labels.json").exists()
    assert "test" in result and result["test"]["n"] == 8
