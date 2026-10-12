"""Prepare KArSL raw-video archives for landmark extraction.

Input (as downloaded from the public KArSL Drive folder):
    data/raw/KARSL-502_Labels.xlsx
    data/raw/<signer>/<train|test>/<from>-<to>.7z        e.g. data/raw/01/train/0071-0170.7z

Output:
    data/karsl_labels.json                               {"0071": {"ar": ..., "en": ...}, ...}
    data/videos/<signID>/<signer>_<split>_<n>.<ext>      layout expected by extract_landmarks.py

    python scripts/prepare_karsl.py --signers 01 --ranges 0071-0170

Extraction uses the 7z executable when present on PATH, otherwise py7zr (slower).
Video files are moved, not copied, so the extracted tree is consumed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
SIGN_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def export_labels(xlsx: Path, dest: Path) -> dict[str, dict[str, str]]:
    import openpyxl

    ws = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    labels: dict[str, dict[str, str]] = {}
    for i, (sid, ar, en) in enumerate(ws.iter_rows(values_only=True)):
        if i == 0 or sid is None:
            continue
        labels[f"{int(sid):04d}"] = {"ar": str(ar).strip(), "en": str(en).strip()}
    dest.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"labels: {len(labels)} signs -> {dest}")
    return labels


def extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("7z") or shutil.which("7za")
    if exe:
        subprocess.run(
            [exe, "x", "-y", f"-o{out_dir}", str(archive)], check=True, stdout=subprocess.DEVNULL
        )
    else:
        import py7zr

        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=out_dir)


def arrange(extracted: Path, signer: str, split: str, videos_dir: Path) -> int:
    """Move every video under `extracted` into videos/<signID>/<signer>_<split>_<n>.<ext>."""
    counters: dict[str, int] = {}
    moved = 0
    for f in sorted(p for p in extracted.rglob("*") if p.suffix.lower() in VIDEO_EXT):
        rel = f.relative_to(extracted)
        # Sign ID: first 4-digit token in the relative path (folder names first, then file name).
        sid = None
        for part in list(rel.parts[:-1]) + [f.stem]:
            m = SIGN_RE.search(part)
            if m:
                sid = m.group(1)
                break
        if sid is None:
            print(f"skip (no sign id): {rel}")
            continue
        n = counters.get(sid, 0)
        counters[sid] = n + 1
        dest = videos_dir / sid / f"s{signer}_{split}_{n:03d}{f.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), dest)
        moved += 1
    print(f"  {signer}/{split}: moved {moved} videos across {len(counters)} signs")
    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--signers", nargs="+", default=["01"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--ranges", nargs="+", default=["0071-0170"])
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    raw, videos = Path(args.raw), Path(args.videos)
    export_labels(raw / "KARSL-502_Labels.xlsx", Path("data/karsl_labels.json"))
    total = 0
    for signer in args.signers:
        for split in args.splits:
            for rng in args.ranges:
                archive = raw / signer / split / f"{rng}.7z"
                if not archive.exists():
                    print(f"missing: {archive}")
                    continue
                work = raw / signer / split / f"_x_{rng}"
                print(f"extracting {archive} ...")
                extract_archive(archive, work)
                total += arrange(work, signer, split, videos)
                shutil.rmtree(work, ignore_errors=True)
                if not args.keep_archives:
                    archive.unlink()
    print(f"done: {total} videos in {videos}")


if __name__ == "__main__":
    main()
