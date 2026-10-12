"""Download the MediaPipe holistic landmarker model (about 14 MB, free) into models/."""

from pathlib import Path
from urllib.request import urlretrieve

from ssl_recognition.landmarks import DEFAULT_MODEL, MODEL_URL

if __name__ == "__main__":
    dest = Path(DEFAULT_MODEL)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already present: {dest}")
    else:
        print(f"downloading {MODEL_URL}")
        urlretrieve(MODEL_URL, dest)
        print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
