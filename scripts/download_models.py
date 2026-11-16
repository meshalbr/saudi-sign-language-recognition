"""Download everything needed to run the app and the demos (free, about 24 MB in total).

    python scripts/download_models.py              # MediaPipe models + trained model
    python scripts/download_models.py --no-run     # MediaPipe models only (for training)

- models/holistic_landmarker.task   MediaPipe body, hands and face (Google, 14 MB)
- models/hand_landmarker.task       MediaPipe hands, used to recover hands MediaPipe loses (8 MB)
- runs/final_v3_face/               trained Transformer (ONNX + PyTorch) from the GitHub release
- data/karsl_labels.json            English and Arabic names of the KArSL signs
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from ssl_recognition.landmarks import DEFAULT_MODEL, HAND_MODEL, HAND_MODEL_URL, MODEL_URL

ROOT = Path(__file__).resolve().parents[1]
RELEASE_URL = (
    "https://github.com/meshalbr/saudi-sign-language-recognition/releases/download/"
    "v1.0/final_v3_face.zip"
)
RUN_DIR = ROOT / "runs" / "final_v3_face"


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"already present: {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urlretrieve(url, dest)
    print(f"saved {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")


def fetch_run() -> None:
    if (RUN_DIR / "model.onnx").exists() and (ROOT / "data" / "karsl_labels.json").exists():
        print(f"already present: {RUN_DIR.relative_to(ROOT)}")
        return
    print(f"downloading {RELEASE_URL}")
    with urlopen(RELEASE_URL) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in archive.namelist():
        target = (ROOT / name).resolve()
        if ROOT.resolve() not in target.parents:  # never write outside the repo
            raise SystemExit(f"unsafe path in archive: {name}")
        if name.endswith("/"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
        print(f"saved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-run", action="store_true", help="skip the trained model")
    args = ap.parse_args()
    fetch(MODEL_URL, Path(DEFAULT_MODEL))
    fetch(HAND_MODEL_URL, Path(HAND_MODEL))
    if not args.no_run:
        fetch_run()
