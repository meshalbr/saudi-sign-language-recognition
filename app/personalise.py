"""Personalisation: fine-tune the deployed model on the user's own recorded signs.

The base model learned from three professional KArSL signers, so a new person has to sign very
precisely. Here a few examples per word from the actual user are mixed with a sample of KArSL
(so the model does not forget the other signers) and the model is fine-tuned for a few epochs on
CPU. The result is written to a new run folder and hot-swapped into the server.
"""

from __future__ import annotations

import json
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from ssl_recognition.data import ClipDataset, load_clip, load_dataset, save_clip
from ssl_recognition.model import build_model

USER_DIR = Path("data/user")
STATUS: dict = {"state": "idle", "epoch": 0, "epochs": 0, "message": ""}
_lock = threading.Lock()


def save_example(sid: str, x: np.ndarray) -> int:
    d = USER_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("*.npz")))
    save_clip(d / f"user_{int(time.time() * 1000)}.npz", x, sid, "user", 30.0)
    return n + 1


def counts() -> dict[str, int]:
    if not USER_DIR.is_dir():
        return {}
    return {d.name: len(list(d.glob("*.npz"))) for d in USER_DIR.iterdir() if d.is_dir()}


def _train(base_run: Path, out_run: Path, epochs: int, on_done) -> None:
    try:
        ckpt = torch.load(base_run / "best.pt", map_location="cpu")
        cfg = dict(ckpt["cfg"])
        label_map = json.loads((base_run / "labels.json").read_text(encoding="utf-8"))
        user = [load_clip(p) for p in sorted(USER_DIR.glob("*/*.npz"))]
        user = [c for c in user if c.label in label_map]
        if not user:
            raise RuntimeError("no recorded examples yet")
        rng = random.Random(0)
        karsl = [c for c in load_dataset(Path(cfg["data_dir"])) if "_train_" in c.path.name]
        karsl = rng.sample(karsl, min(2500, len(karsl)))
        seq = int(cfg.get("seq_len", 32))
        reps = max(1, 600 // len(user))  # oversample the user's few examples
        ds = ConcatDataset(
            [ClipDataset(karsl, label_map, seq, train=True, cfg=cfg)]
            + [ClipDataset(user, label_map, seq, train=True, cfg=cfg, seed=i) for i in range(reps)]
        )
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        model = build_model(cfg, ckpt["num_classes"])
        model.load_state_dict(ckpt["state_dict"])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        STATUS.update(state="training", epoch=0, epochs=epochs, message="")
        for ep in range(1, epochs + 1):
            model.train()
            for x, y in dl:
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            STATUS["epoch"] = ep
        # accuracy on the user's own examples (training data, so a sanity check, not a score)
        model.eval()
        with torch.no_grad():
            ud = DataLoader(ClipDataset(user, label_map, seq, cfg=cfg), batch_size=64)
            hit = sum(int((model(x).argmax(1) == y).sum()) for x, y in ud)
        # Build the new run in a staging folder so a failed export never leaves a mixed run.
        stage = out_run.with_name(out_run.name + "_staging")
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        torch.save(
            {"state_dict": model.state_dict(), "cfg": cfg, "num_classes": ckpt["num_classes"]},
            stage / "best.pt",
        )
        for name in ("labels.json", "config.json"):
            if (base_run / name).exists():
                shutil.copy(base_run / name, stage / name)
        _export(stage)
        shutil.rmtree(out_run, ignore_errors=True)
        stage.rename(out_run)
        on_done(out_run)
        STATUS.update(
            state="done",
            message=f"trained on {len(user)} of your examples; fits {hit}/{len(user)} of them",
        )
    except Exception as e:  # noqa: BLE001 - report any failure to the page
        STATUS.update(state="error", message=str(e))


def _export(run: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:  # works however the server was started
        sys.path.insert(0, str(root))
    from scripts.export_onnx import export

    export(run)


def start(base_run: Path, out_run: Path, on_done, epochs: int = 4) -> bool:
    with _lock:
        if STATUS["state"] == "training":
            return False
        STATUS.update(state="training", epoch=0, epochs=epochs, message="starting")
    threading.Thread(target=_train, args=(base_run, out_run, epochs, on_done), daemon=True).start()
    return True
