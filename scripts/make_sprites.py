"""Render the reference video strip of every sign and pack them for the GitHub release.

    python scripts/make_sprites.py              # data/sprites/<sign>_<clip>.jpg for all signs
    python scripts/make_sprites.py --zip sprites.zip

Each strip is the app's /api/sprite image: the best-tracked signer 01 test clip of a sign (the
same clip as the skeleton reference), 24 frames side by side, square crop. Needs the KArSL videos
in data/videos/<sign>/. The zip is uploaded to the v1.0 release, and download_models.py fetches it
so the GitHub Pages demo can show the real signer next to the skeleton.

Reference videos come from the KArSL dataset (KFUPM): Sidig, Luqman, Mahmoud and Mohandes,
"KArSL: Arabic Sign Language Database", ACM TALLIP 20(1), 2021.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", help="also write this zip (paths relative to the repo root)")
    args = ap.parse_args()
    os.chdir(ROOT)  # the server resolves data/ relative to the working directory
    sys.path.insert(0, str(ROOT))
    from app import server

    server.STATE.update(run="runs/final_v3_face", names="data/karsl_labels.json")
    server.predictor()  # loads the run and fills STATE["labels"], which sprite() checks
    sids = [lab["id"] for lab in server.STATE["labels"]]
    made = []
    for sid in sids:
        stem = server._best_reference_stem(sid)
        if stem is None:
            print(f"{sid}: no reference landmarks, skipped")
            continue
        server.sprite(sid)
        dest = Path("data/sprites") / f"{sid}_{stem}_{server.SPRITE_H}.jpg"
        if dest.exists():
            made.append(dest)
        else:
            print(f"{sid}: no video for {stem}, skipped")
    size = sum(p.stat().st_size for p in made) / 1e6
    print(f"{len(made)} sprites in data/sprites ({size:.1f} MB)")
    if args.zip:
        # stored, not deflated: the JPEGs are already compressed
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_STORED) as z:
            for p in made:
                z.write(p, p.as_posix())
        print(f"wrote {args.zip}")


if __name__ == "__main__":
    main()
