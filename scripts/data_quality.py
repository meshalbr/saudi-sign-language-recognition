"""Per-sign landmark quality report for KArSL: how often MediaPipe lost the hands.

python scripts/data_quality.py            # writes reports/data_quality.md
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np


def gaps(present: np.ndarray) -> list[int]:
    """Lengths of runs of missing frames strictly inside the clip (hands seen before and after)."""
    idx = np.where(present)[0]
    if len(idx) < 2:
        return []
    out, run = [], 0
    for v in present[idx[0] : idx[-1] + 1]:
        if not v:
            run += 1
        elif run:
            out.append(run)
            run = 0
    return out


def main() -> None:
    names = json.loads(Path("data/karsl_labels.json").read_text(encoding="utf-8"))
    rows = collections.defaultdict(
        lambda: {"cov": [], "gap_clips": 0, "n": 0, "max_gap": 0, "long": 0}
    )
    for f in glob.glob("data/landmarks/*/*.npz"):
        sid = f.replace("\\", "/").split("/")[-2]
        x = np.load(f)["x"]
        present = (x[:, -2] + x[:, -1]) > 0
        g = gaps(present)
        r = rows[sid]
        r["n"] += 1
        r["cov"].append(present.mean())
        r["gap_clips"] += bool(g)
        r["long"] += any(v > 8 for v in g)
        r["max_gap"] = max(r["max_gap"], max(g, default=0))
    table = []
    for sid, r in rows.items():
        table.append(
            (
                sid,
                names.get(sid, {}).get("en", ""),
                names.get(sid, {}).get("ar", ""),
                float(np.mean(r["cov"])),
                r["gap_clips"] / r["n"],
                r["long"] / r["n"],
                r["max_gap"],
                r["n"],
            )
        )
    table.sort(key=lambda t: t[3])
    lines = [
        "# Landmark quality per sign (KArSL-100, MediaPipe Holistic)",
        "",
        "Coverage = share of frames with at least one hand detected. Gap = hands lost mid-sign "
        "(seen before and after). Long gap = more than 8 frames (0.27 s).",
        "",
        "| Sign | English | Arabic | Hand coverage | Clips with a gap | Clips with a long gap | "
        "Longest gap (frames) | Clips |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sid, en, ar, cov, g, lg, mg, n in table:
        lines.append(f"| {sid} | {en} | {ar} | {cov:.0%} | {g:.0%} | {lg:.0%} | {mg} | {n} |")
    Path("reports/data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    covs = np.array([t[3] for t in table])
    print(
        f"signs: {len(table)} | coverage <90%: {(covs < 0.9).sum()} | 90-95%: "
        f"{((covs >= 0.9) & (covs < 0.95)).sum()} | >=95%: {(covs >= 0.95).sum()}"
    )
    print("worst 15:")
    for sid, en, _, cov, g, lg, mg, _ in table[:15]:
        print(
            f"  {sid} {en[:24]:24s} coverage {cov:.0%}  gap-clips {g:.0%}  "
            f"long-gap-clips {lg:.0%}  max {mg}"
        )


if __name__ == "__main__":
    main()
