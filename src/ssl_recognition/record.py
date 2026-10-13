"""Record your own sign clips from the webcam as landmark .npz files.

    python -m ssl_recognition.record --label hello --signer meshal --clips 10 --frames 40

Controls: SPACE starts a clip (after a 3-2-1 countdown), Q quits. Each clip stores only the
155-d landmark features per frame, so 10 clips are a few hundred kilobytes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .data import save_clip
from .landmarks import HolisticExtractor, draw_result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="sign name, e.g. hello or مرحبا")
    ap.add_argument("--signer", required=True, help="who is signing, e.g. meshal")
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--frames", type=int, default=40, help="frames per clip")
    ap.add_argument("--out", default="data/landmarks")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out) / args.label
    existing = len(list(out_dir.glob(f"{args.signer}_*.npz"))) if out_dir.exists() else 0
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    recorded = 0
    with HolisticExtractor() as ex:
        t_start = time.time()
        while recorded < args.clips:
            ok, frame = cap.read()
            if not ok:
                break
            # Landmark the raw frame (training videos are not mirrored); mirror only for display.
            feat = ex.process(frame, int((time.time() - t_start) * 1000))
            draw_result(frame, ex.last_result)
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"{args.label}  clip {recorded + 1}/{args.clips}  SPACE=record  Q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            hands = int(feat[-2] + feat[-1])
            cv2.putText(
                frame, f"hands: {hands}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("record", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue

            # Countdown, then capture `frames` consecutive frames.
            for n in (3, 2, 1):
                t_end = time.time() + 0.6
                while time.time() < t_end:
                    ok, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(
                        frame, str(n), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6
                    )
                    cv2.imshow("record", frame)
                    cv2.waitKey(1)
            feats = []
            while len(feats) < args.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                feats.append(ex.process(frame, int((time.time() - t_start) * 1000)))
                draw_result(frame, ex.last_result)
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    f"REC {len(feats)}/{args.frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("record", frame)
                cv2.waitKey(1)
            x = np.stack(feats)
            idx = existing + recorded
            path = out_dir / f"{args.signer}_{idx:03d}.npz"
            save_clip(path, x, args.label, args.signer, fps)
            recorded += 1
            print(
                f"saved {path}  frames={len(x)}  hands_seen={int((x[:, -2] + x[:, -1] > 0).sum())}"
            )

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {recorded} clips for '{args.label}'")


if __name__ == "__main__":
    main()
