import torch

from ssl_recognition.landmarks import FEATURE_DIM
from ssl_recognition.model_transformer import TransformerClassifier


def test_transformer_forward_shape():
    m = TransformerClassifier(num_classes=7, in_dim=FEATURE_DIM, dim=32, heads=4, layers=1)
    out = m(torch.zeros(3, 32, FEATURE_DIM))
    assert out.shape == (3, 7)


def test_transformer_exports_to_onnx(tmp_path):
    m = TransformerClassifier(num_classes=5, in_dim=FEATURE_DIM, dim=32, heads=4, layers=1).eval()
    dest = tmp_path / "t.onnx"
    torch.onnx.export(m, torch.zeros(1, 32, FEATURE_DIM), str(dest), opset_version=17, dynamo=False)
    assert dest.exists() and dest.stat().st_size > 1000
