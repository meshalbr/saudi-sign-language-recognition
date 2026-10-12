"""Extract landmark features from a folder of sign videos (for KArSL RGB or any dataset).

Expected input layout:  <videos>/<label>/<signer>_<anything>.mp4
Output layout:          <out>/<label>/<signer>_<anything>.npz

    python scripts/extract_landmarks.py --videos data/videos --out data/landmarks --signer-from-name

Run this once on Colab for big datasets; the resulting .npz files are small enough for the laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ssl_recognition.data import save_clip
from ssl_recognition.landmarks import HolisticExtractor

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def extract_video(ex: HolisticExtractor, path: Path, max_frames: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    feats = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= max_frames:
            break
        feats.append(ex.process(frame, int(i * 1000 / fps)))
        i += 1
    cap.release()
    if not feats:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(feats), fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/landmarks")
    ap.add_argument("--signer", default=None, help="fixed signer name for every clip")
    ap.add_argument(
        "--signer-from-name", action="store_true", help="signer = text before the first '_'"
    )
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.videos).glob("*/*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"no videos under {args.videos}")
    out = Path(args.out)
    done = skipped = failed = 0
    with HolisticExtractor() as ex:
        for v in tqdm(videos, unit="video"):
            label = v.parent.name
            signer = args.signer or (v.stem.split("_")[0] if args.signer_from_name else "unknown")
            dest = out / label / f"{v.stem}.npz"
            if dest.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                x, fps = extract_video(ex, v, args.max_frames)
            except Exception as e:  # noqa: BLE001 - keep going over a big dataset
                failed += 1
                print(f"FAILED {v}: {e}")
                continue
            save_clip(dest, x, label, signer, fps)
            done += 1
            # VIDEO mode timestamps must keep increasing across files.
            ex._last_ts += 10_000
    print(f"extracted={done} skipped={skipped} failed={failed} -> {out}")


if __name__ == "__main__":
    main()
