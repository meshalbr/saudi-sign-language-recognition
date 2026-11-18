"""Build the static, server-free version of the app for GitHub Pages.

    python scripts/download_models.py        # models, trained run, reference skeletons
    python scripts/build_site.py --out site  # then serve site/ with any static file server

Everything the page normally asks the server for becomes a plain file at the same relative path
(api/labels, api/info, api/ref/<sign>), and app/static/engine.js runs the preprocessing, the
sign segmenter and the ONNX model in the browser (ONNX Runtime Web). Nothing leaves the visitor's
device. tests/test_engine_parity.py keeps engine.js identical to the Python code.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ssl_recognition.features import fill_gaps

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
ORT_WEB = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
SPRITE_H = 240  # same as app.server.SPRITE_H (part of the sprite file name)


def coverage(npz: Path) -> float:
    with np.load(npz) as z:
        x = z["x"]
    return float(((x[:, -2] + x[:, -1]) > 0).mean())


def best_clip(sid: str, landmarks: Path) -> Path | None:
    """Same choice as app.server._best_reference_stem: best-tracked signer 01 test clip."""
    d = landmarks / sid
    cands = sorted(d.glob("s01_test_*.npz")) or sorted(d.glob("*.npz"))
    return max(cands[:20], key=coverage) if cands else None


def reference(clip: Path, sid: str) -> dict:
    """Same format as app.server.reference."""
    with np.load(clip) as z:
        x = fill_gaps(z["x"], 8)
    def pts(row, a, b, n):
        return np.round(row[a:b].reshape(n, 3)[:, :2], 3).tolist()

    frames = [
        {
            "pose": pts(row, 0, 27, 9),
            "left": pts(row, 27, 90, 21) if row[153] > 0 else [],
            "right": pts(row, 90, 153, 21) if row[154] > 0 else [],
        }
        for row in x
    ]
    return {"id": sid, "fps": 30, "frames": frames}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/final_v3_face")
    ap.add_argument("--names", default="data/karsl_labels.json")
    ap.add_argument("--landmarks", default="data/landmarks")
    ap.add_argument("--sprites", default="data/sprites")
    ap.add_argument("--out", default="site")
    args = ap.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    if out.exists():
        if any(out.iterdir()) and not (out / ".nojekyll").exists():
            raise SystemExit(f"{out} is not empty and not an earlier site build; not deleting it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    labels = json.loads((run / "labels.json").read_text(encoding="utf-8"))
    table = json.loads((ROOT / args.names).read_text(encoding="utf-8"))
    names = []
    for lab in labels:
        sid = str(lab).split()[0]
        row = table.get(sid, {})
        names.append({"id": sid, "ar": row.get("ar", ""), "en": row.get("en", sid)})

    # the page, with the in-browser engine switched on
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    marker = '<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/'
    assert html.count(marker) == 1, "index.html layout changed; update build_site.py"
    inject = (
        "<script>window.SSL_STATIC=true</script>\n"
        f'<script src="{ORT_WEB}"></script>\n'
        '<script src="engine.js"></script>\n'
    )
    (out / "index.html").write_text(html.replace(marker, inject + marker), encoding="utf-8")
    shutil.copy(STATIC / "engine.js", out / "engine.js")

    # model and data the server would otherwise provide
    (out / "model").mkdir()
    shutil.copy(run / "model.onnx", out / "model" / "model.onnx")
    write_json(out / "model" / "config.json", cfg)
    write_json(out / "api" / "labels", names)
    write_json(
        out / "api" / "info",
        {"public": True, "static": True, "camera_url": None, "displays": 0,
         "hand_recovery": bool(cfg.get("hand_recovery", False))},
    )
    write_json(
        out / "api" / "user",
        {"counts": {}, "status": {"state": "disabled"}, "model": "base", "public": True},
    )
    # reference skeleton per sign, plus the real reference video strip (KArSL, signer 01) when
    # download_models.py has fetched the sprites (scripts/make_sprites.py makes them)
    n_refs = n_vids = 0
    for n in names:
        clip = best_clip(n["id"], ROOT / args.landmarks)
        if clip is None:
            continue
        write_json(out / "api" / "ref" / n["id"], reference(clip, n["id"]))
        n_refs += 1
        strip = ROOT / args.sprites / f"{n['id']}_{clip.stem}_{SPRITE_H}.jpg"
        if strip.exists():
            (out / "api" / "sprite").mkdir(parents=True, exist_ok=True)
            shutil.copy(strip, out / "api" / "sprite" / f"{n['id']}.jpg")
            n_vids += 1
    (out / "models").mkdir()
    for task in ("holistic_landmarker.task", "hand_landmarker.task"):
        shutil.copy(ROOT / "models" / task, out / "models" / task)
    (out / ".nojekyll").write_text("")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"built {out}: {len(names)} signs, {n_refs} reference skeletons, "
          f"{n_vids} reference videos, {size:.1f} MB")


if __name__ == "__main__":
    main()
