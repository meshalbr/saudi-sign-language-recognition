"""Extract landmark features from a folder of sign videos (for KArSL RGB or any dataset).

Expected input layout:  <videos>/<label>/<signer>_<anything>.mp4
Output layout:          <out>/<label>/<signer>_<anything>.npz

    python scripts/extract_landmarks.py --videos data/videos --out data/landmarks \
        --signer-from-name --workers 2 --width 640

Frames are downscaled to ``--width`` before MediaPipe (it resizes internally anyway, and
1080p decoding plus colour conversion dominates otherwise). ``--workers`` runs that many
independent extractor processes; on a 4-core laptop 2 is the sweet spot.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from ssl_recognition.data import save_clip
from ssl_recognition.landmarks import HolisticExtractor

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def extract_video(
    ex: HolisticExtractor, path: Path, max_frames: int, width: int
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    feats = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= max_frames:
            break
        if width and frame.shape[1] > width:
            h = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
        feats.append(ex.process(frame, int(i * 1000 / fps)))
        i += 1
    cap.release()
    if not feats:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(feats), fps


def _signer_of(v: Path, fixed: str | None, from_name: bool) -> str:
    if fixed:
        return fixed
    return v.stem.split("_")[0] if from_name else "unknown"


def _worker(job: tuple) -> tuple[int, int, int, list[str]]:
    """Process one chunk of videos with its own extractor. Returns (done, skipped, failed, errs)."""
    videos, out, signer, from_name, max_frames, width, overwrite, wid, min_conf = job
    done = skipped = failed = 0
    errs: list[str] = []
    t0 = time.time()
    with HolisticExtractor(min_conf=min_conf) as ex:
        for k, v in enumerate(videos, 1):
            dest = out / v.parent.name / f"{v.stem}.npz"
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            try:
                x, fps = extract_video(ex, v, max_frames, width)
                save_clip(dest, x, v.parent.name, _signer_of(v, signer, from_name), fps)
                done += 1
            except Exception as e:  # noqa: BLE001 - keep going over a big dataset
                failed += 1
                errs.append(f"{v}: {e}")
            ex._last_ts += 10_000  # VIDEO mode timestamps must keep increasing across files
            if k % 100 == 0:
                rate = k / (time.time() - t0)
                print(f"[w{wid}] {k}/{len(videos)}  {rate * 60:.0f} videos/min", flush=True)
    return done, skipped, failed, errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/landmarks")
    ap.add_argument("--signer", default=None, help="fixed signer name for every clip")
    ap.add_argument(
        "--signer-from-name", action="store_true", help="signer = text before the first '_'"
    )
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--width", type=int, default=640, help="downscale frames to this width")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.5, help="MediaPipe detection confidence")
    ap.add_argument("--signs", nargs="*", default=None, help="only these sign folders")
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.videos).glob("*/*") if p.suffix.lower() in VIDEO_EXT)
    if args.signs:
        keep = set(args.signs)
        videos = [v for v in videos if v.parent.name in keep]
    if not videos:
        raise SystemExit(f"no videos under {args.videos}")
    out = Path(args.out)
    print(f"{len(videos)} videos, {args.workers} worker(s), width {args.width}", flush=True)
    t0 = time.time()
    chunks = [videos[i :: args.workers] for i in range(args.workers)]
    jobs = [
        (
            c,
            out,
            args.signer,
            args.signer_from_name,
            args.max_frames,
            args.width,
            args.overwrite,
            i,
            args.min_conf,
        )
        for i, c in enumerate(chunks)
    ]
    if args.workers == 1:
        results = [_worker(jobs[0])]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
    done = sum(r[0] for r in results)
    skipped = sum(r[1] for r in results)
    failed = sum(r[2] for r in results)
    for r in results:
        for e in r[3][:20]:
            print("FAILED", e)
    mins = (time.time() - t0) / 60
    print(f"extracted={done} skipped={skipped} failed={failed} in {mins:.1f} min -> {out}")


if __name__ == "__main__":
    main()
