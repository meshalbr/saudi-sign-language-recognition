"""Build the README demo GIF (and an MP4) from KArSL videos of a signer the model never saw.

    python scripts/make_demo.py --run runs/p6_s02_face --signer s02

Three panels per frame: the camera with the skeleton drawn on it, the skeleton alone (what
the model actually receives), and the live top-3 prediction, updated as the sign unfolds,
exactly like the webcam app (growing window, prediction every few frames).
The run must be a leave-one-signer-out model whose test signer is --signer, so every clip
in the demo comes from a person the model has never seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ssl_recognition.infer import Predictor
from ssl_recognition.landmarks import HolisticExtractor, _to_array

try:  # Arabic shaping for Pillow builds without raqm
    import arabic_reshaper
    from bidi.algorithm import get_display

    def ar(s: str) -> str:
        return get_display(arabic_reshaper.reshape(s))
except ImportError:  # pragma: no cover
    def ar(s: str) -> str:
        return s

FONT_DIR = Path("C:/Windows/Fonts")
S = 340  # panel size (square)
HEAD, FOOT = 46, 46
BG, PANEL, INK, MUTED = (17, 24, 39), (31, 41, 55), (243, 244, 246), (156, 163, 175)
GREEN, AMBER, BAR = (52, 211, 153), (251, 191, 36), (55, 65, 81)
HAND_C = (52, 211, 153)
HAND2_C = (244, 114, 182)
POSE_C = (96, 165, 250)
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for n in (name, "segoeui.ttf", "arial.ttf"):
        p = FONT_DIR / n
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_TITLE = font("seguisb.ttf", 19)
F_SMALL = font("segoeui.ttf", 14)
F_BIG = font("seguisb.ttf", 26)
F_MED = font("segoeui.ttf", 17)
F_AR = font("tahoma.ttf", 18)
F_AR_BIG = font("tahomabd.ttf", 24)


def crop_box(pose: np.ndarray, w: int, h: int) -> tuple[int, int, int]:
    """Square crop around the shoulders: (x0, y0, size) in pixels."""
    size = int(h * 0.9)
    cx = (pose[11, 0] + pose[12, 0]) / 2 * w
    x0 = int(np.clip(cx - size / 2, 0, w - size))
    return x0, h - size, size


def to_px(arr, box, w, h):
    x0, y0, size = box
    return [((x * w - x0) / size * S, (y * h - y0) / size * S) for x, y, _ in arr]


def draw_skeleton(d: ImageDraw.ImageDraw, pose, left, right, box, w, h, dx=0, r=3):
    if pose is not None:
        p = to_px(pose, box, w, h)
        for a, b in POSE_EDGES:
            d.line([(p[a][0] + dx, p[a][1]), (p[b][0] + dx, p[b][1])], fill=POSE_C, width=3)
        for i in list(range(0, 11)) + [11, 12, 13, 14, 15, 16]:
            d.ellipse([p[i][0] + dx - r, p[i][1] - r, p[i][0] + dx + r, p[i][1] + r], fill=POSE_C)
    for hand, col in ((left, HAND_C), (right, HAND2_C)):
        if hand is None:
            continue
        q = to_px(hand, box, w, h)
        for a, b in HAND_EDGES:
            d.line([(q[a][0] + dx, q[a][1]), (q[b][0] + dx, q[b][1])], fill=col, width=2)
        for x, y in q:
            d.ellipse([x + dx - 2, y - 2, x + dx + 2, y + 2], fill=col)


def render_clip(video: Path, pred: Predictor, names: dict, truth: str, stride: int, every: int):
    cap = cv2.VideoCapture(str(video))
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA))
    cap.release()
    h, w = raw[0].shape[:2]
    feats, faces, skel, top = [], [], [], []
    with HolisticExtractor() as ex:
        for i, f in enumerate(raw):
            feats.append(ex.process(f, int(i * 1000 / 30)))
            faces.append(ex.last_face.copy())
            res = ex.last_result
            pose = _to_array(res.pose_landmarks, 33)
            left = _to_array(res.left_hand_landmarks, 21)
            right = _to_array(res.right_hand_landmarks, 21)
            left = left if left is not None else ex.recovered.get("left")
            right = right if right is not None else ex.recovered.get("right")
            skel.append((pose, left, right))
    box = next(crop_box(p, w, h) for p, _, _ in skel if p is not None)

    frames, current = [], []
    n = len(raw)
    for i in range(n):
        if i + 1 >= 10 and ((i + 1) % stride == 0 or i == n - 1):
            p = pred.predict(np.stack(feats[: i + 1]), np.stack(faces[: i + 1]))
            order = np.argsort(-p)[:3]
            current = [(pred.labels[k], float(p[k])) for k in order]
        if i % every == 0 or i == n - 1:
            frames.append(frame_image(raw[i], skel[i], box, w, h, current, names, truth, i + 1, n,
                                      final=False))
    final = frame_image(raw[-1], skel[-1], box, w, h, current, names, truth, n, n, final=True)
    top = current
    return frames, final, top


def frame_image(bgr, sk, box, w, h, top, names, truth, i, n, final):
    x0, y0, size = box
    crop = cv2.cvtColor(bgr[y0:y0 + size, x0:x0 + size], cv2.COLOR_BGR2RGB)
    W, H = 3 * S + 4 * 12, S + HEAD + FOOT
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # header
    d.text((12, 12), "Saudi Sign Language recognition", font=F_TITLE, fill=INK)
    tag = "signer never seen in training · CPU only"
    d.text((W - 12 - d.textlength(tag, font=F_SMALL), 16), tag, font=F_SMALL, fill=MUTED)
    # panel 1: camera + skeleton
    xa = 12
    img.paste(Image.fromarray(crop).resize((S, S), Image.BILINEAR), (xa, HEAD))
    pose, left, right = sk
    lay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw_skeleton(ImageDraw.Draw(lay), pose, left, right, box, w, h)
    img.paste(lay, (xa, HEAD), lay)
    # panel 2: what the model sees
    xb = xa + S + 12
    d.rectangle([xb, HEAD, xb + S, HEAD + S], fill=PANEL)
    lay2 = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw_skeleton(ImageDraw.Draw(lay2), pose, left, right, box, w, h, r=4)
    img.paste(lay2, (xb, HEAD), lay2)
    d.text((xb + 10, HEAD + 8), "What the model sees", font=F_MED, fill=INK)
    d.text((xb + 10, HEAD + 30), "body, hand and face points only", font=F_SMALL, fill=MUTED)
    # panel 3: prediction
    xc = xb + S + 12
    d.rectangle([xc, HEAD, xc + S, HEAD + S], fill=PANEL)
    d.text((xc + 14, HEAD + 10), "Live prediction", font=F_MED, fill=INK)
    # progress through the sign
    d.rectangle([xc + 14, HEAD + 38, xc + S - 14, HEAD + 42], fill=BAR)
    d.rectangle([xc + 14, HEAD + 38, xc + 14 + int((S - 28) * i / n), HEAD + 42], fill=MUTED)
    if not top:
        d.text((xc + 14, HEAD + 70), "watching the hands…", font=F_MED, fill=MUTED)
    else:
        y = HEAD + 58
        for k, (lab, prob) in enumerate(top):
            en, arn = names[lab]["en"], names[lab]["ar"]
            col = (GREEN if lab == truth else AMBER) if k == 0 else MUTED
            fb = F_BIG if k == 0 else F_MED
            fa = F_AR_BIG if k == 0 else F_AR
            d.text((xc + 14, y), en[:1].upper() + en[1:], font=fb, fill=INK if k == 0 else MUTED)
            at = ar(arn)
            d.text((xc + S - 14 - d.textlength(at, font=fa), y + 2), at, font=fa,
                   fill=INK if k == 0 else MUTED)
            y += 36 if k == 0 else 26
            bw = S - 28 - 50
            d.rounded_rectangle([xc + 14, y, xc + 14 + bw, y + 12], 6, fill=BAR)
            fill_w = max(12, int(bw * prob))
            d.rounded_rectangle([xc + 14, y, xc + 14 + fill_w, y + 12], 6, fill=col)
            d.text((xc + S - 14 - d.textlength(f"{prob * 100:.0f}%", font=F_SMALL), y - 3),
                   f"{prob * 100:.0f}%", font=F_SMALL, fill=col)
            y += 28
        if final:
            ok = top[0][0] == truth
            msg = "Correct: the true sign" if ok else f"Wrong: true sign is {names[truth]['en']}"
            d.rounded_rectangle([xc + 14, HEAD + S - 52, xc + S - 14, HEAD + S - 14], 8,
                                fill=(6, 78, 59) if ok else (120, 53, 15))
            yb = HEAD + S - 33  # vertical centre of the badge
            if ok:  # draw the tick: the font has no check-mark glyph
                d.line([(xc + 28, yb), (xc + 34, yb + 6), (xc + 46, yb - 7)], fill=GREEN, width=4)
            else:
                d.line([(xc + 28, yb - 7), (xc + 42, yb + 7)], fill=AMBER, width=4)
                d.line([(xc + 28, yb + 7), (xc + 42, yb - 7)], fill=AMBER, width=4)
            d.text((xc + 56, HEAD + S - 45), msg, font=F_MED, fill=INK)
    # footer: the pipeline
    steps = ["camera", "MediaPipe skeleton", "Transformer (450k params)", "sign"]
    x, yf = 12, HEAD + S + 13
    for k, s in enumerate(steps):
        d.text((x, yf), s, font=F_SMALL, fill=INK if k in (0, 3) else MUTED)
        x += d.textlength(s, font=F_SMALL) + 8
        if k < len(steps) - 1:
            d.text((x, yf), "→", font=F_SMALL, fill=MUTED)
            x += 22
    t = "2 MB ONNX model · 1.6 ms per prediction"
    d.text((W - 12 - d.textlength(t, font=F_SMALL), yf), t, font=F_SMALL, fill=MUTED)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/p6_s02_face")
    ap.add_argument("--signer", default="s02")
    ap.add_argument("--signs", default="0088,0092,0117,0160,0162,0132,0161,0115")
    ap.add_argument("--count", type=int, default=5, help="how many correct clips to keep")
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--every", type=int, default=2, help="keep every n-th frame (30 fps source)")
    ap.add_argument("--stride", type=int, default=3, help="predict every n frames")
    ap.add_argument("--colors", type=int, default=96, help="GIF palette size")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if cfg.get("test_signer") != args.signer:
        raise SystemExit(f"{run} was not trained without {args.signer}; the demo would be unfair")
    names = json.loads(Path("data/karsl_labels.json").read_text(encoding="utf-8"))
    pred = Predictor(run)

    gif, kept = [], []
    for sign in args.signs.split(","):
        clips = sorted(Path("data/videos", sign).glob(f"{args.signer}_test_*.mp4"))
        if not clips:
            continue
        frames, final, top = render_clip(clips[0], pred, names, sign, args.stride, args.every)
        ok = bool(top) and top[0][0] == sign
        print(f"{sign} {names[sign]['en']:<12} {clips[0].name}  top1={top[0][0] if top else '-'}"
              f" p={top[0][1] if top else 0:.2f}  {'kept' if ok else 'skipped'}")
        if ok:
            gif += [(f, 33 * args.every) for f in frames] + [(final, 1400)]
            kept.append(sign)
        if len(kept) >= args.count:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mc, nd = Image.Quantize.MEDIANCUT, Image.Dither.NONE
    imgs = [im.quantize(colors=args.colors, method=mc, dither=nd) for im, _ in gif]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=[t for _, t in gif],
                 loop=0, optimize=True, disposal=1)
    mp4 = out.with_suffix(".mp4")
    w, h = gif[0][0].size
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for im, t in gif:
        fr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        for _ in range(max(1, round(t * 15 / 1000))):
            vw.write(fr)
    vw.release()
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB, {mp4} {mp4.stat().st_size / 1e6:.1f} MB, "
          f"signs kept: {kept}")


if __name__ == "__main__":
    main()
